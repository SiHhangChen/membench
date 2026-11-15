from dataclasses import replace
from pathlib import Path

from bench.evaluation.suite import load_eval_suite
from bench.tasks.wr12.evaluation import REPORT_STAGE_ORDER, REPORT_STAGE_PROFILE_ID

ROOT = Path(__file__).resolve().parents[2]


def test_wr12_act_dp_share_physical_protocol_and_three_report_stages():
    act = load_eval_suite(ROOT / "configs/eval/suites/wr12_v1.yaml")
    dp = load_eval_suite(ROOT / "configs/eval/diffusion/wr12_r1_v1.yaml")
    assert act.task_id == dp.task_id == "WR-12"
    assert act.seeds == dp.seeds == tuple(range(1050, 1100))
    assert act.task_options == dp.task_options
    assert act.env_split == dp.env_split
    assert act.artifacts == dp.artifacts
    assert act.artifacts.record_video
    assert act.transport == dp.transport
    assert act.rollout.max_steps == 2400
    assert act.rollout.replan_steps == 50
    assert dp.rollout == replace(act.rollout, replan_steps=1)
    assert act.expert_prefix is dp.expert_prefix is None
    assert REPORT_STAGE_PROFILE_ID == "WR-12/report-v2"
    assert REPORT_STAGE_ORDER == ("block_picked", "block_placed", "success")
