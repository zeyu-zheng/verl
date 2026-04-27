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

"""Compatibility patches for ``vllm.utils.system_utils`` on Hopper hosts.

vLLM's ``CudaRTLibrary`` resolves ``libcudart`` via ``find_loaded_library``,
which scans ``/proc/self/maps`` and returns the first matching path. Once
``tilelang`` is imported (lazily by ``fla``'s TileLangBackend dispatch on
Hopper), it dlopens its bundled ``libcudart_stub.so``, which then shadows the
real CUDA runtime in that scan and causes ``undefined symbol:
cudaDeviceReset`` deep inside vLLM's CUDA initialisation. ``tilelang`` ships
the stub baked into the wheel with no opt-out, so we filter it out before vLLM
ever reads ``find_loaded_library``.
"""

_PATCH_MARKER = "_verl_tilelang_stub_filter_applied"

# Modules in vLLM that bind ``find_loaded_library`` at import time via
# ``from vllm.utils.system_utils import find_loaded_library``. Patching only
# ``system_utils`` is insufficient for these because they captured the function
# object at their own import time; we have to rebind the attribute on each
# already-imported caller too. This list must stay in sync with vLLM's tree.
_FIND_LOADED_LIBRARY_CALLERS = (
    "vllm.distributed.device_communicators.cuda_wrapper",
    "vllm.device_allocator.cumem",
)


def apply_find_loaded_library_patch() -> None:
    """Wrap ``vllm.utils.system_utils.find_loaded_library`` to skip stubs.

    Best-effort: silently no-ops when vLLM is not installed. Idempotent: if
    the source module is already patched we still re-walk caller modules
    below in case any of them was imported after the first patch invocation
    (e.g. when the teacher rollout pulls in ``fla -> tilelang`` between
    ``CudaRTLibrary()`` calls in a colocated worker setup).
    """
    import sys

    try:
        import vllm.utils.system_utils as system_utils
    except ImportError:
        return

    if getattr(system_utils, _PATCH_MARKER, False):
        _rebind_in_callers(sys, system_utils.find_loaded_library)
        return

    original_find_loaded_library = system_utils.find_loaded_library

    def find_loaded_library(lib_name: str):
        try:
            with open("/proc/self/maps") as fp:
                matches = []
                for line in fp:
                    if lib_name not in line:
                        continue
                    try:
                        start = line.index("/")
                    except ValueError:
                        continue
                    matches.append(line[start:].strip())
        except OSError:
            return original_find_loaded_library(lib_name)

        for path in matches:
            filename = path.rsplit("/", 1)[-1]
            if "_stub" in filename or "/tilelang/" in path or "/tvm/" in path:
                continue
            if filename.rpartition(".so")[0].startswith(lib_name):
                return path
        return original_find_loaded_library(lib_name)

    system_utils.find_loaded_library = find_loaded_library
    setattr(system_utils, _PATCH_MARKER, True)
    _rebind_in_callers(sys, find_loaded_library)


def _rebind_in_callers(sys_module, patched_fn) -> None:
    """Rebind ``find_loaded_library`` in any vLLM module that already imported
    the original via ``from vllm.utils.system_utils import find_loaded_library``.

    Ignores modules we have not yet imported -- they will pick up the patched
    function the first time they execute their own import statement.
    """
    for mod_name in _FIND_LOADED_LIBRARY_CALLERS:
        mod = sys_module.modules.get(mod_name)
        if mod is None:
            continue
        if getattr(mod, "find_loaded_library", None) is patched_fn:
            continue
        mod.find_loaded_library = patched_fn
