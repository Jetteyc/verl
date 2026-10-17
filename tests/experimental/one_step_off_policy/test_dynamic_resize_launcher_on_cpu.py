# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""The two versions of ``shell/run_dynamic_resize.sh``: what the launcher runs, without running it.

A stand-in ``python3`` at the front of ``PATH`` records every command the launcher starts, and where and
in which environment it starts it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "verl/experimental/one_step_off_policy/shell/run_dynamic_resize.sh"
OPTIMIZER_CHECK = "tests/special_distributed/run_fsdp2_resize_optimizer.py"

STAND_IN_PYTHON3 = f"""#!{sys.executable}
import json, os, sys

if sys.argv[1:] == ["-"]:  # the data conversion script on stdin: hand the launcher its two paths back
    print("TRAIN_FILE=train.parquet")
    print("VAL_FILE=val.parquet")
    sys.exit(0)
with open(os.environ["RECORD"], "a") as record:
    record.write(json.dumps({{"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()}}) + "\\n")
if sys.argv[1:3] == ["-m", "torch.distributed.run"]:
    sys.exit(int(os.environ.get("OPTIMIZER_CHECK_EXIT_CODE", "0")))
"""


@pytest.fixture
def launch(tmp_path):
    """Run the launcher with ``env`` on top of a bare environment; its exit code and the commands it started."""
    stand_in = tmp_path / "bin" / "python3"
    stand_in.parent.mkdir()
    stand_in.write_text(STAND_IN_PYTHON3)
    stand_in.chmod(0o755)
    record = tmp_path / "record.jsonl"

    def run(*args, **env):
        record.unlink(missing_ok=True)
        env = {
            "PATH": f"{stand_in.parent}:{os.environ['PATH']}",
            "RECORD": str(record),
            "CKPT_DIR": str(tmp_path / "ckpt"),
            "MODEL_PATH": "/models/some-lm",
            **env,
        }
        result = subprocess.run(["bash", str(LAUNCHER), *args], env=env, capture_output=True, text=True, timeout=120)
        commands = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
        return result.returncode, commands

    run.trace = [
        "trainer.dynamic_resize.trace.enable=true",
        f"trainer.dynamic_resize.trace.path={tmp_path}/ckpt/resize_trace.jsonl",
    ]

    run.ckpt_dir = str(tmp_path / "ckpt")
    return run


def _is_training(command):
    return command["argv"][:2] == ["-m", "verl.experimental.one_step_off_policy.main_ppo"]


def test_plain_version_only_starts_training(launch):
    code, commands = launch("trainer.total_training_steps=3")

    assert code == 0
    (training,) = commands
    assert _is_training(training)
    assert training["argv"][-1] == "trainer.total_training_steps=3"
    assert not [arg for arg in training["argv"] if "dynamic_resize.trace" in arg]
    assert "VERL_LOGGING_LEVEL" not in training["env"]


def test_verification_version_checks_the_optimizer_then_trains_with_every_resize_log(launch):
    _, (plain,) = launch("trainer.total_training_steps=3")
    code, commands = launch("trainer.total_training_steps=3", VERL_RESIZE_VERIFY="1")

    assert code == 0
    check, training = commands
    # before training: the optimizer check, on the model and through the actor sizes of this launch
    assert check["argv"] == ["-m", "torch.distributed.run", "--standalone", "--nproc-per-node=6", OPTIMIZER_CHECK]
    assert (REPO_ROOT / OPTIMIZER_CHECK).is_file()
    assert check["cwd"] == str(REPO_ROOT)
    assert check["env"]["PYTHONPATH"].split(":")[0] == str(REPO_ROOT)
    assert check["env"]["MODEL_PATH"] == "/models/some-lm"
    assert check["env"]["RESIZE_SEQUENCE"] == "4,6,2"
    assert check["env"]["OUT_DIR"] == f"{launch.ckpt_dir}/optimizer_check"
    # then the training of the plain version, in which the workers verify every switch, every resize line
    # is logged and every step is traced; what the caller appends still comes last
    assert training["argv"] == plain["argv"][:-1] + launch.trace + ["trainer.total_training_steps=3"]
    assert training["env"]["VERL_RESIZE_VERIFY"] == "1"
    assert training["env"]["VERL_LOGGING_LEVEL"] == "DEBUG"
    assert "RESIZE_SEQUENCE" not in training["env"]


@pytest.mark.parametrize(
    ("env", "sequence", "processes"),
    [
        ({"NUM_GPUS": "4", "N_GPUS_ROLLOUT": "2", "SPLIT_PLAN": "[3,1]", "SECOND_SPLIT_PLAN": "[1,3]"}, "2,3,1", 3),
        ({"SECOND_SPLIT_PLAN": "[4,4]"}, "4,6,4", 6),
        ({"N_GPUS_ROLLOUT": "2", "FIRST_SPLIT_PLAN": "[2,6]", "SECOND_SPLIT_PLAN": "[4,4]"}, "6,2,4", 6),
    ],
)
def test_optimizer_check_goes_through_the_actor_sizes_of_the_schedule(launch, env, sequence, processes):
    code, (check, training) = launch(VERL_RESIZE_VERIFY="1", **env)

    assert code == 0
    assert check["env"]["RESIZE_SEQUENCE"] == sequence
    assert f"--nproc-per-node={processes}" in check["argv"]
    assert _is_training(training)


def test_failed_optimizer_check_stops_the_launcher_before_training(launch):
    code, commands = launch(VERL_RESIZE_VERIFY="1", OPTIMIZER_CHECK_EXIT_CODE="1")

    assert code != 0
    assert [command["argv"][-1] for command in commands] == [OPTIMIZER_CHECK]


@pytest.mark.parametrize("env", [{"HOST_STAGING_BACKEND": "disk_fallback"}, {"OPTIMIZER_CHECK_ENABLE": "false"}])
def test_verification_version_without_the_optimizer_check_keeps_the_logs(launch, env):
    code, commands = launch(VERL_RESIZE_VERIFY="1", **env)

    assert code == 0
    (training,) = commands
    assert _is_training(training)
    assert training["env"]["VERL_LOGGING_LEVEL"] == "DEBUG"
    assert training["argv"][-2:] == launch.trace


def test_verification_version_keeps_a_logging_level_the_caller_chose(launch):
    code, commands = launch(VERL_RESIZE_VERIFY="1", VERL_LOGGING_LEVEL="INFO")

    assert code == 0
    assert [command["env"]["VERL_LOGGING_LEVEL"] for command in commands] == ["INFO", "INFO"]
