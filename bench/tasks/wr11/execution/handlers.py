"""Expert-only action orchestration; evaluation uses physical evidence alone."""

import numpy as np

from bench.execution.manipulation.object_interactions import get_object_hold_status
from bench.execution.navigation.base_motion import qpos_for_controller
from bench.execution.planning import object_reference_pos


def build_operation_handlers(controller):
    def observe_sample(target, **kwargs):
        del target
        if not controller.runtime_config.execute:
            return {"pass": True, "dry_run": True}
        right = controller.env.robots[0].composite_controller.part_controllers["right"]
        controller.driver.run_gripper(
            arm_qpos=qpos_for_controller(controller.env, right),
            gripper_action=-1.0,
            steps=6,
            post_step_callback=kwargs.get("post_step_callback"),
        )
        controller.refresh_obs()
        result = controller.env.task_diagnostics()
        passed = bool(result["stage_checks"]["sample_observed"])
        return {
            "primitive": "observe_sample",
            "pass": passed,
            "diagnostics": result,
            "reason": None if passed else "sample_not_visible_at_observation_station",
        }

    def place(item, **kwargs):
        env = controller.env
        position = env.delivery_pose()
        yaws = kwargs.pop("placement_yaw_offsets_deg", [0, 90, 180, 270])

        def pre_release():
            hold = get_object_hold_status(env, item)
            bottom = object_reference_pos(env, item, placement_reference="bbox_bottom_center")
            checks = {
                "held": bool(hold["held"]),
                "near_delivery_xy": bool(np.linalg.norm(bottom[:2] - position[:2]) < 0.14),
                "near_delivery_z": bool(abs(bottom[2] - position[2]) < 0.08),
            }
            return {"pass": all(checks.values()), "checks": checks}

        return controller.place_at_world_pose_candidates(
            item,
            candidate_groups=[
                {
                    "group_name": "wr11_delivery_yaws",
                    "candidates": [
                        {
                            "target_pos_world": position.tolist(),
                            "placement_yaw_offset_deg": float(yaw),
                        }
                        for yaw in yaws
                    ],
                }
            ],
            target_name="delivery_tray",
            pre_release_check=pre_release if controller.runtime_config.execute else None,
            **kwargs,
        )

    def verify(item, **kwargs):
        if not controller.runtime_config.execute:
            return {"pass": True, "dry_run": True}
        right = controller.env.robots[0].composite_controller.part_controllers["right"]
        controller.driver.run_gripper(
            arm_qpos=qpos_for_controller(controller.env, right),
            gripper_action=-1.0,
            steps=6,
            post_step_callback=kwargs.get("post_step_callback"),
        )
        passed = bool(controller.env._check_success())
        return {
            "pass": passed,
            "primitive": "verify_delivery",
            "object_name": item,
            "reason": None if passed else "delivery_not_physically_confirmed",
        }

    return {"observe_sample": observe_sample, "place_delivery": place, "verify_delivery": verify}
