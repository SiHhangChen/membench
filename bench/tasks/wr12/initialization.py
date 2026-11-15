"""Reset the robot at the observation station before recorded actions begin."""

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic


def initialize_episode_state(descriptor, env, observation):
    plan = descriptor.episode_plan
    env.configure_episode(plan)
    xy = np.array(plan.base_xy_by_station["observe_station"])
    yaw = plan.base_yaw_by_station["observe_station"]
    observation = set_base_pose_kinematic(env, target_xy=xy, target_yaw=yaw)
    env.reset_evaluation()
    return observation, {
        "initial_base_xy": xy.tolist(),
        "initial_base_yaw": yaw,
        "sampling_profile_id": plan.sampling_profile_id,
        "setup_recorded_in_hdf5_actions": False,
    }
