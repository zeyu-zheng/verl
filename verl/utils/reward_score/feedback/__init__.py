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

"""Reward functions for formal-math (Lean 4) RL.

Self-contained: shells out to ``lake exe repl`` directly, no external
prover repo required. Toolchain discovery honours ``FORMAL_LAKE_PATH`` /
``FORMAL_LEAN_WORKSPACE``; see :mod:`.formal_math` for defaults.
"""

from verl.utils.reward_score.feedback import formal_math  # noqa: F401
