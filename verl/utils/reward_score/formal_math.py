# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Formal-math (Lean 4) reward via an in-process Kimina REPL pool.

Drives ``kimina-lean-server``'s asyncio ``Manager`` directly from the reward
worker -- no uvicorn server, no HTTP. The pool of ``lake env repl``
subprocesses is created once on first call (``initialize_repls`` warms
``max_repls`` Mathlib-loaded REPLs in ~30s) and reused for the whole
training run; each REPL serves up to ``LEAN_SERVER_MAX_REPL_USES`` checks
before being recycled.

Both the per-sample ``compute_score`` (consumed by
``NaiveRewardManager`` / the ``default_compute_score`` dispatcher) and the
batched ``compute_score_batch`` (consumed by ``BatchRewardManager``) route
through the same warm pool; the batched entry fans the whole ``train_batch``
out via a single ``asyncio.gather`` so verification is bound by
``ceil(batch / max_repls) * avg_compile_time`` instead of a serial loop.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)


_LEAN_TOOLCHAIN_FAILURE = (
    "Cannot locate kimina-lean-server. Set KIMINA_LEAN_SERVER_ROOT or run "
    "the trainer from the FBI repo root (so ``./kimina-lean-server`` resolves)."
)


def _bootstrap_kimina() -> Path:
    """Resolve kimina-lean-server, put it on sys.path, prime its env vars.

    Lookup order:
      1. ``KIMINA_LEAN_SERVER_ROOT`` env var.
      2. ``$CWD/kimina-lean-server`` (canonical FBI layout, all launchers cd
         to FBI root before invoking python).
      3. ``<this file>/../../../../../kimina-lean-server`` (verl as an FBI
         submodule).

    The hosted ``Settings`` reads ``LEAN_SERVER_*`` from ``$CWD/.env``;
    when invoked from outside ``kimina-lean-server/`` we manually load the
    pinned ``.env`` so ``settings.repl_path`` / ``settings.project_dir`` /
    ``settings.max_repls`` etc. agree with the operator-vetted values.
    """
    candidates: list[Path] = []
    env_root = os.environ.get("KIMINA_LEAN_SERVER_ROOT")
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(Path.cwd() / "kimina-lean-server")
    # ${FBI}/verl/verl/utils/reward_score/formal_math.py.parents[4] == ${FBI}
    candidates.append(Path(__file__).resolve().parents[4] / "kimina-lean-server")

    for cand in candidates:
        manager_py = cand / "server" / "manager.py"
        if not manager_py.is_file():
            continue
        cand = cand.resolve()
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        env_file = cand / ".env"
        if env_file.is_file():
            try:
                from dotenv import load_dotenv

                load_dotenv(env_file, override=False)
            except ImportError:
                logger.warning("python-dotenv missing; skipping kimina .env load")
        return cand

    raise RuntimeError(_LEAN_TOOLCHAIN_FAILURE)


_KIMINA_ROOT = _bootstrap_kimina()

from kimina_client import ReplResponse, Snippet  # noqa: E402
from server.manager import Manager  # noqa: E402
from server.split import split_snippet  # noqa: E402

# ---------------------------------------------------------------------------
# Persistent asyncio loop on a daemon thread (sync -> async bridge)
# ---------------------------------------------------------------------------

_loop: Optional[asyncio.AbstractEventLoop] = None
_loop_thread: Optional[threading.Thread] = None
_loop_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return the singleton background asyncio loop.

    Reward workers are invoked from sync code paths (`NaiveRewardManager.__call__`
    et al), so the asyncio glue lives in a daemon thread; coroutines are
    submitted via ``asyncio.run_coroutine_threadsafe`` and joined on the
    caller. The loop is created exactly once per process and lives until
    interpreter shutdown -- this is what lets the kimina ``Manager`` keep
    its 64-REPL pool warm across the whole training run.
    """
    global _loop, _loop_thread
    if _loop is not None:
        return _loop
    with _loop_lock:
        if _loop is not None:
            return _loop
        loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            try:
                loop.run_forever()
            finally:
                loop.close()

        thread = threading.Thread(target=_run, name="kimina-asyncio", daemon=True)
        thread.start()
        _loop = loop
        _loop_thread = thread
        return _loop


def _run_coro(coro: Any) -> Any:
    """Submit a coroutine to the daemon loop and block until it returns."""
    return asyncio.run_coroutine_threadsafe(coro, _get_loop()).result()


# ---------------------------------------------------------------------------
# Manager singleton (warm REPL pool, lifetime = process)
# ---------------------------------------------------------------------------

_manager: Optional[Manager] = None
_manager_lock = threading.Lock()


def _get_manager() -> Manager:
    """Lazy-init a single ``Manager`` and ``initialize_repls`` once.

    Subsequent reward calls reuse the same warm pool -- the ~30s Mathlib
    import per REPL is paid exactly once, then every check goes through
    ``Manager._free`` -> ``Repl.send_timeout`` against an already-imported
    process. With ``LEAN_SERVER_MAX_REPL_USES=200`` each REPL in the pool
    serves up to 200 checks before being recycled, so the warmup amortises
    over thousands of verifications per pool refresh cycle.
    """
    global _manager
    if _manager is not None:
        return _manager
    with _manager_lock:
        if _manager is not None:
            return _manager
        # Manager.__init__ kwargs default to server.settings.settings.{max_repls, ...}
        # which were populated from kimina-lean-server/.env during _bootstrap_kimina().
        mgr = Manager()
        _run_coro(mgr.initialize_repls())
        _manager = mgr
        return mgr


# ---------------------------------------------------------------------------
# Lean code extraction / assembly (legacy helpers, behaviour unchanged)
# ---------------------------------------------------------------------------


def extract_lean_code(solution_str: str) -> str:
    """Return the last fenced ``lean4`` (or ``lean``) block in ``solution_str``.

    Tries ``lean4`` first since it is a prefix of ``lean`` and would otherwise
    yield duplicate hits inside the same fence. Falls back to the trimmed full
    string if no fenced block is present.
    """
    for marker in ("```lean4", "```lean"):
        if marker not in solution_str:
            continue
        blocks = [part.split("```", 1)[0].strip() for part in solution_str.split(marker)[1:]]
        if blocks:
            return blocks[-1]
    return solution_str.strip()


def assemble_full_code(formal_statement: str, proof_body: str) -> str:
    """Splice ``proof_body`` onto the canonical statement at ``:= by``.

    If the model already emitted a complete file (starts with ``import``),
    return it unchanged; otherwise concatenate ``<statement up to := by>`` +
    ``proof_body``. Falls back to whichever side is non-empty.
    """
    if proof_body and proof_body.lstrip().startswith("import "):
        return proof_body
    if formal_statement and ":= by" in formal_statement and proof_body:
        return formal_statement.split(":= by")[0] + ":= by\n" + proof_body
    return proof_body or formal_statement


# ---------------------------------------------------------------------------
# Score translation
# ---------------------------------------------------------------------------


def _to_score_dict(response: ReplResponse, verify_time: float) -> dict:
    """Translate ``ReplResponse`` into verl's reward-extra dict.

    Mirrors the field set produced by the previous lake-subprocess path so
    metric aggregation in the reward manager keeps working unchanged.
    """
    error = getattr(response, "error", None)
    payload = getattr(response, "response", None) or {}

    sorries = payload.get("sorries", []) or []
    messages = payload.get("messages", []) or []
    errors = [m.get("data", "") for m in messages if m.get("severity") == "error"]
    sorry_warnings = [
        m.get("data", "")
        for m in messages
        if m.get("severity") == "warning"
        and ("declaration uses 'sorry'" in m.get("data", "") or "failed" in m.get("data", ""))
    ]

    passed = error is None and not errors
    complete = passed and not sorries and not sorry_warnings

    feedback_parts: list[str] = []
    if errors:
        feedback_parts.append("Lean errors: " + "; ".join(errors[:3]))
    if sorries:
        feedback_parts.append("Proof contains sorry.")
    if error:
        feedback_parts.append(str(error))

    score = float(complete)
    return {
        "score": score,
        "acc": score,
        "pass": float(passed),
        "complete": float(complete),
        "feedback": " ".join(p for p in feedback_parts if p),
        "verify_time": verify_time,
    }


def _build_snippet(idx: int, solution_str: str, extra_info: dict) -> Snippet:
    formal_statement = extra_info.get("formal_statement") or extra_info.get("formalization", "")
    code = assemble_full_code(formal_statement, extract_lean_code(solution_str))
    return Snippet(id=str(idx), code=code)


def _shift_line(pos, offset: int) -> None:
    if not pos:
        return
    line = pos.get("line")
    if line is not None:
        pos["line"] = line + offset


def _apply_header_offset(response: ReplResponse, offset: int) -> None:
    """Re-anchor REPL message line numbers from body-relative to file-relative.

    Mirrors ``server.routers.check._apply_header_offset``; inlined to avoid
    pulling fastapi onto the import path of the reward function.
    """
    if offset <= 0 or response.error is not None:
        return
    payload = response.response
    if not payload:
        return
    messages = payload.get("messages")
    if not messages:
        return
    for message in messages:
        _shift_line(message.get("pos"), offset)
        _shift_line(message.get("endPos"), offset)


async def _check_one(manager: Manager, snippet: Snippet, timeout: float) -> ReplResponse:
    """Single-snippet REPL check. Mirrors the in-process slice of
    ``server.routers.check.run_checks.run_one`` minus FastAPI/Prisma/DB calls."""
    repl = None
    try:
        split = split_snippet(snippet.code)
        try:
            repl = await manager.get_repl(split.header, snippet.id, reuse=True)
            prep = await manager.prep(repl, snippet.id, timeout, debug=False)
            if prep is not None and prep.error:
                return prep
            resp = await repl.send_timeout(
                Snippet(id=snippet.id, code=split.body), timeout, infotree=None
            )
            _apply_header_offset(resp, split.header_line_count)
            await manager.release_repl(repl)
            resp.diagnostics = None
            return resp
        except TimeoutError:
            uuid_hex = repl.uuid.hex if repl is not None else ""
            if repl is not None:
                await manager.destroy_repl(repl)
            return ReplResponse(
                id=snippet.id,
                error=f"Lean REPL command timed out in {timeout} seconds",
                time=timeout,
                diagnostics={"repl_uuid": uuid_hex} if uuid_hex else None,
            )
        except Exception as exc:
            uuid_hex = repl.uuid.hex if repl is not None else ""
            if repl is not None:
                await manager.destroy_repl(repl)
            logger.exception("Snippet %s execution failed", snippet.id)
            return ReplResponse(
                id=snippet.id,
                error=f"REPL execution failed: {exc}",
                time=0.0,
                diagnostics={"repl_uuid": uuid_hex} if uuid_hex else None,
            )
    except asyncio.CancelledError:
        if repl is not None:
            await manager.destroy_repl(repl)
        raise


async def _run_checks(
    manager: Manager, snippets: list[Snippet], timeout: float
) -> list[ReplResponse]:
    return list(await asyncio.gather(*(_check_one(manager, s, timeout) for s in snippets)))


# ---------------------------------------------------------------------------
# Public sync entry points
# ---------------------------------------------------------------------------


def compute_score(
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict] = None,
    **_: Any,
) -> dict:
    """Single-sample reward used by ``NaiveRewardManager`` and the dispatcher.

    Wraps a 1-element batch around ``compute_score_batch`` so the reward
    backend is single-sourced; ``NaiveRewardManager`` will see the same
    ``{score, acc, pass, complete, feedback, verify_time}`` payload as the
    batched path.
    """
    out = compute_score_batch(
        data_sources=["formal_math"],
        solution_strs=[solution_str],
        ground_truths=[ground_truth],
        extra_infos=[extra_info or {}],
    )
    return out[0]


def compute_score_batch(
    data_sources: Iterable[str],
    solution_strs: list[str],
    ground_truths: list,
    extra_infos: list[dict],
    **_: Any,
) -> list[dict]:
    """Batch reward used by ``BatchRewardManager`` (parallel verification).

    Fans every snippet at the warm Manager via a single ``asyncio.gather``
    inside ``_run_checks``; the pool's ``max_repls`` cap acts as the natural
    concurrency limit (extra snippets queue on the Manager's
    ``asyncio.Condition`` until a REPL frees up).
    """
    del data_sources  # all rows are formal_math by construction
    extras = [e or {} for e in extra_infos]
    if len(solution_strs) != len(extras):
        raise ValueError(
            f"compute_score_batch: solution_strs ({len(solution_strs)}) and "
            f"extra_infos ({len(extras)}) length mismatch"
        )
    if not solution_strs:
        return []

    snippets = [
        _build_snippet(i, s, e) for i, (s, e) in enumerate(zip(solution_strs, extras, strict=True))
    ]
    timeout = max((int(e.get("timeout", 300)) for e in extras), default=300)

    mgr = _get_manager()  # warmup happens here on first call
    start = time.time()
    try:
        responses = _run_coro(_run_checks(mgr, snippets, float(timeout)))
    except Exception:
        err = traceback.format_exc()
        logger.exception("formal_math compute_score_batch failed")
        return [
            {
                "score": 0.0,
                "acc": 0.0,
                "pass": 0.0,
                "complete": 0.0,
                "feedback": f"reward backend error: {err.splitlines()[-1] if err else 'unknown'}",
                "verify_time": 0.0,
            }
            for _ in snippets
        ]

    elapsed = time.time() - start
    per_call = elapsed / max(len(responses), 1)
    return [_to_score_dict(r, per_call) for r in responses]
