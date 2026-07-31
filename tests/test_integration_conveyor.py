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
    # Added after a user-reported visual grasp failure: `grasped`/
    # `grasp_error_m` alone (a commit-instant distance check) were never
    # sufficient proof of an actual pick -- this repo's own earlier state
    # passed both of the assertions above while the object was never really
    # held. `contact_verified` is a real, direct check (both fingers
    # simultaneously in contact with the object, MuJoCo's own contact
    # array) -- see design spec Section 12 and sim/conveyor_scene.py's
    # `is_grasped()`.
    assert result["contact_verified"] is True
