"""Place the robot at the sample station once, before the action rollout."""

import numpy as np

from bench.execution.navigation.base_motion import set_base_pose_kinematic


def initialize_episode_state(descriptor, env, observation):
    env.configure_episode(descriptor.episode_plan)
    plan = descriptor.episode_plan
    xy = np.array(plan.base_xy_by_station["sample_station"])
    observation = set_base_pose_kinematic(env, target_xy=xy, target_yaw=plan.base_yaw)
    env.reset_evaluation()
    return observation, {
        "initial_base_xy": xy.tolist(),
        "initial_base_yaw": plan.base_yaw,
        "sampling_profile_id": plan.sampling_profile_id,
        "base_xy_by_station": dict(plan.base_xy_by_station),
        "base_radius_by_station": dict(plan.base_radius_by_station),
        "sample_station_reused_for_return": True,
        "setup_recorded_in_hdf5_actions": False,
    }
