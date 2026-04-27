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
# Adapted from https://github.com/deepseek-ai/DeepSeek-Prover-V1.5

import json
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Optional

# Override via FORMAL_LAKE_PATH / FORMAL_LEAN_WORKSPACE; both default to the
# elan + REPL layout produced by `curl https://elan.lean-lang.org/elan-init.sh`.
DEFAULT_LAKE_PATH = str(Path.home() / ".elan/bin/lake")
DEFAULT_LEAN_WORKSPACE = "mathlib4/"


def resolve_lake_path(extra_info: Optional[dict] = None) -> Optional[str]:
    """Resolve the ``lake`` binary in priority order: hint -> env -> PATH -> default."""
    extra_info = extra_info or {}
    for candidate in (
        extra_info.get("lake_path"),
        os.environ.get("FORMAL_LAKE_PATH"),
        shutil.which("lake"),
        DEFAULT_LAKE_PATH,
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def resolve_lean_workspace(extra_info: Optional[dict] = None) -> str:
    """Resolve the Lean workspace (mathlib4 checkout) in priority order."""
    extra_info = extra_info or {}
    return (
        extra_info.get("lean_workspace")
        or os.environ.get("FORMAL_LEAN_WORKSPACE")
        or DEFAULT_LEAN_WORKSPACE
    )


def _build_lake_env(lake_path: str) -> dict:
    """Inject the elan toolchain's lib dir into LD_LIBRARY_PATH."""
    env = os.environ.copy()
    toolchain_lib = os.path.join(os.path.dirname(os.path.dirname(lake_path)), "lib")
    if os.path.isdir(toolchain_lib):
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{toolchain_lib}:{existing}" if existing else toolchain_lib
    return env


def verify_lean4_file(code: str, lake_path: str, lean_workspace: str, timeout: int = 300) -> dict:
    """Run ``lake exe repl`` on ``code`` and parse the JSON response.

    Mirrors the minimal subset of DeepSeek-Prover-V1.5's verifier that verl's
    reward router needs: ``pass`` / ``complete`` / ``errors`` / ``sorries`` /
    ``system_errors`` / ``verify_time``. Drops the AST / tactics / premises
    branches since they are not consumed by ``compute_score``.
    """
    command = json.dumps({"cmd": code}, ensure_ascii=False)
    start_time = time.time()
    try:
        with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as temp_file:
            temp_file.write(command + "\r\n\r\n")
            temp_file.seek(0)
            outputs = subprocess.run(
                [lake_path, "exe", "repl"],
                stdin=temp_file,
                capture_output=True,
                text=True,
                cwd=lean_workspace,
                timeout=timeout,
                env=_build_lake_env(lake_path),
                check=False,
            )
        result = json.loads(outputs.stdout)
        messages = result.get("messages", [])
        warnings = [m for m in messages if m.get("severity") == "warning"]
        result = {
            "sorries": result.get("sorries", []),
            "errors": [m for m in messages if m.get("severity") == "error"],
            "warnings": warnings,
            "system_errors": None,
        }
        result["pass"] = not result["errors"]
        result["complete"] = (
            result["pass"]
            and not result["sorries"]
            and not any(
                "declaration uses 'sorry'" in w.get("data", "") or "failed" in w.get("data", "")
                for w in warnings
            )
        )
    except Exception:
        result = {"pass": False, "complete": False, "system_errors": traceback.format_exc()}
    result["verify_time"] = time.time() - start_time
    return result


def extract_lean_code(solution_str: str) -> str:
    """Return the last ```lean4``` (or ```lean```) block in ``solution_str``.

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


def compute_score(solution_str: str, ground_truth: str, extra_info: Optional[dict] = None) -> dict:
    """Verify a Lean 4 proof against its formal statement.

    The formal statement (``theorem ... := by``) lives on
    ``extra_info['formal_statement']``; we splice the model's proof body onto
    its ``:= by`` and forward the result to ``lake exe repl``. Returns a dict
    with ``score`` / ``acc`` / ``pass`` / ``complete`` / ``feedback`` /
    ``verify_time`` so it slots into verl's reward router.
    """
    extra_info = extra_info or {}
    timeout = int(extra_info.get("timeout", 300))
    formal_statement = extra_info.get("formal_statement") or extra_info.get("formalization", "")
    lake_path = resolve_lake_path(extra_info)
    lean_workspace = resolve_lean_workspace(extra_info)

    code = assemble_full_code(formal_statement, extract_lean_code(solution_str))

    if lake_path is None:
        return {
            "score": 0.0,
            "acc": 0.0,
            "pass": 0.0,
            "complete": 0.0,
            "feedback": "Lean toolchain not found. Set FORMAL_LAKE_PATH or install `lake`.",
            "verify_time": 0.0,
        }

    result = verify_lean4_file(
        code=code, lake_path=lake_path, lean_workspace=lean_workspace, timeout=timeout
    )
    reward = float(result.get("complete", False))
    feedback_parts = []
    if result.get("errors"):
        feedback_parts.append("Lean errors: " + "; ".join(err.get("data", "") for err in result["errors"][:3]))
    if result.get("sorries"):
        feedback_parts.append("Proof contains sorry.")
    if result.get("system_errors"):
        feedback_parts.append(str(result["system_errors"]))
    return {
        "score": reward,
        "acc": reward,
        "pass": float(bool(result.get("pass", False))),
        "complete": float(bool(result.get("complete", False))),
        "feedback": " ".join(part for part in feedback_parts if part),
        "verify_time": result.get("verify_time", 0.0),
    }
