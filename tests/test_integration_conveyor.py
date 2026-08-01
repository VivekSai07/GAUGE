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
    # Added to close a real gap: the old contact_verified check ran
    # immediately after the gripper closed, while the object was still
    # sitting where it was grasped -- that proves contact, not a hold. A
    # gripper closed around an object resting on the platform, without
    # enough grip to support it once airborne, would have passed the old
    # check. contact_verified is now checked after a real ~10cm lift (see
    # run_conveyor_demo.py).
    #
    # object_peak_height_gain_m (not object_height_gain_m) is the right
    # proof-of-lift check here: object_height_gain_m is the FINAL gain after
    # the full lift+settle, so an object that was genuinely carried upward
    # and then slipped back down before the window ends reads as a small or
    # even negative number -- indistinguishable from "never lifted" -- while
    # object_peak_height_gain_m records the highest gain seen at any point
    # during the lift+settle, so it stays a reliable positive signal that a
    # real lift happened even if the grip didn't hold.
    assert result["object_peak_height_gain_m"] > 0.05
