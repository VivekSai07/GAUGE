import numpy as np
import yaml

from run_conveyor_demo import run_one_episode


def test_conveyor_episode_grasps_within_tolerance():
    with open("configs/conveyor.yaml") as f:
        config = yaml.safe_load(f)

    result = run_one_episode(config)

    assert result["grasped"] is True
    assert result["grasp_error_m"] is not None
    assert result["grasp_error_m"] <= config["grasp"]["position_tolerance"] + 0.01
    assert result["steps"] < config["max_steps"]
