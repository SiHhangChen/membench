from types import SimpleNamespace

import numpy as np
import pytest

from bench.tasks.ts02.execution import completion_button


@pytest.mark.parametrize("failure", [None, "retreat", "arm_return"])
def test_each_press_returns_arm_only_after_clearing_button(monkeypatch, failure):
    calls = []
    depth = [0.0]
    eef = np.eye(4)
    button_xyz = np.array([3.85, -2.06, 0.92])
    lock_base = lambda obs: obs
    monkeypatch.setattr(completion_button, "matrix_from_base_obs", lambda obs: np.eye(4))
    monkeypatch.setattr(completion_button, "pose_from_matrix", lambda pose: pose)
    monkeypatch.setattr(
        completion_button, "locked_base_post_step_callback", lambda env: lock_base
    )

    def run_stage(target, **kwargs):
        phase = kwargs["stage"].removeprefix("ts02_completion_button_")
        calls.append(phase)
        eef[:] = target
        depth[0] = 0.006 if phase == "press" else 0.0
        kwargs["post_step_callback"]({})
        return {"pass": phase != failure}

    def hold(**kwargs):
        calls.append("hold")
        assert depth[0] == 0.006
        kwargs["post_step_callback"]({})
        return {"steps": kwargs["steps"], "obs": {}}

    def return_arm(**kwargs):
        calls.append("arm_return")
        # The arm must first clear the plunger vertically before returning home.
        np.testing.assert_allclose(eef[:3, 3], button_xyz + [0, 0, 0.08])
        assert depth[0] == 0.0
        assert kwargs["gripper_action"] == -1.0
        assert kwargs["stage"] == "ts02_post_button_return_to_regular_pose"
        assert kwargs["post_step_callback"]({}) == {}
        return {"pass": failure != "arm_return", "execution": {"executed_steps": 20}}

    controller = SimpleNamespace(
        runtime_config=SimpleNamespace(execute=True),
        env=SimpleNamespace(
            completion_button_position=lambda: button_xyz,
            completion_button_press_depth=lambda: depth[0],
        ),
        obs={},
        refresh_obs=lambda: None,
        _sync_obs=lambda obs: None,
        stage_executor=SimpleNamespace(
            run=run_stage, obs={}, current_eef_site_world=lambda: eef.copy()
        ),
        driver=SimpleNamespace(current_arm_qpos=lambda: np.zeros(7), run_gripper=hold),
        return_arm_to_regular_pose=return_arm,
    )
    for _ in range(2 if failure is None else 1):
        result = completion_button.press_completion_button_physically(
            controller,
            prepress_z_offset=0.08,
            press_z_offset=-0.010,
            hold_steps=12,
            gripper_action=1.0,
            near_miss_position_tolerance=0.09,
            press_depth_threshold=0.004,
        )
        assert result["pass"] is (failure is None)
        assert result["checks"]["arm_returned"] is (failure is None)
        expected = ["prepress", "press", "hold", "retreat"]
        if failure != "retreat":
            expected.append("arm_return")
        assert calls == expected
        if failure is None:
            assert result["stages"]["arm_return"]["execution"]["executed_steps"] == 20
        calls.clear()
