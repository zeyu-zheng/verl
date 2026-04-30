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


def apply_find_loaded_library_patch() -> None:
    """Wrap ``vllm.utils.system_utils.find_loaded_library`` to skip stubs.

    Best-effort: silently no-ops when vLLM is not installed or already
    patched. Must run before any caller binds ``find_loaded_library`` from
    ``vllm.utils.system_utils``.
    """
    try:
        import vllm.utils.system_utils as system_utils
    except ImportError:
        return

    if getattr(system_utils, _PATCH_MARKER, False):
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
