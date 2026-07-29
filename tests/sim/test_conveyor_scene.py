import numpy as np
from sim.conveyor_scene import ConveyorSceneEnv


def test_scene_loads_and_steps():
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    initial_pos = env.get_object_ground_truth().copy()
    for _ in range(50):
        env.step(qdot_cmd=np.zeros(7))
    moved_pos = env.get_object_ground_truth()
    assert moved_pos[1] > initial_pos[1]  # object moved along +y as scripted
    assert env.get_joint_positions().shape == (7,)


def test_rgbd_shapes():
    env = ConveyorSceneEnv(conveyor_velocity=np.array([0.0, 0.1, 0.0]))
    env.reset()
    rgb, depth = env.get_rgbd(width=64, height=64)
    assert rgb.shape == (64, 64, 3)
    assert depth.shape == (64, 64)
    assert depth.dtype == np.float32
