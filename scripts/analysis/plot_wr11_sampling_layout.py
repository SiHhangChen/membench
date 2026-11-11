#!/usr/bin/env python3
"""Draw WR-11's configured station disks and object sampling support."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.tasks.wr11.config import load_task_config, resolve_configured_episode_plan
from bench.tasks.wr11.domain.scene import (
    DELIVERY_TRAY_XY,
    SAMPLE_TRAY_XY,
    STOCK_TRAY_XYS,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Output file stem")
    args = parser.parse_args()
    config = load_task_config()
    stations = json.loads(config.path(config.resources["stations"]).read_text())["stations"]
    rules = json.loads(config.path(config.resources["sampling_rules"]).read_text())["entries"]
    plans = [resolve_configured_episode_plan(config, seed) for seed in range(200)]
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False, "svg.fonttype": "none"})
    blue, orange, green = "#1764BD", "#D96A14", "#14805E"
    ink, muted = "#183248", "#536B7A"
    fig = plt.figure(figsize=(14, 9.6), facecolor="white")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.12, 1], hspace=0.38, wspace=0.30)
    fig.subplots_adjust(top=0.87, bottom=0.13, left=0.075, right=0.965)
    fig.suptitle("WR-11 站位与采样范围", fontsize=23, color=ink, y=0.983, weight="bold")
    fig.text(
        0.5,
        0.919,
        "机器人基座：两个半径 5 cm 的圆盘  ·  牛奶盒位置：各自半径 2.5 cm 的圆盘",
        ha="center",
        fontsize=13,
        color=muted,
    )
    ax = fig.add_subplot(grid[0, :])
    ax.set_xlim(0.50, 4.05)
    ax.set_ylim(-2.42, -1.23)
    ax.set_aspect("equal")
    # The background and tray rectangles are schematic; circle centers and
    # radii use configured world coordinates and the same x/y scale.
    ax.add_patch(
        FancyBboxPatch(
            (0.54, -2.36),
            3.46,
            0.38,
            boxstyle="round,pad=0.01,rounding_size=0.025",
            facecolor="#EEF1F2",
            edgecolor="#C4CDD3",
            linewidth=1.1,
        )
    )
    ax.text(2.17, -2.31, "同一长岛台 · 台面和托盘轮廓为示意", ha="center", color=muted, fontsize=10)
    object_radius = float(rules["objects"]["radius_m"])
    trays = [
        ("样品", SAMPLE_TRAY_XY, True),
        ("送回空托盘", DELIVERY_TRAY_XY, False),
        *[(label, xy, True) for label, xy in zip(("候选左", "候选中", "候选右"), STOCK_TRAY_XYS)],
    ]
    for label, xy, has_object in trays:
        x, y = xy
        ax.add_patch(
            FancyBboxPatch(
                (x - 0.125, y - 0.075),
                0.25,
                0.15,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                facecolor="white",
                edgecolor="#94A5B0",
                linewidth=1,
            )
        )
        if has_object:
            ax.add_patch(
                Circle(xy, object_radius, edgecolor=green, facecolor="#B5E1CF", linewidth=1.8)
            )
            ax.plot(x, y, "+", color=green, markersize=5)
        ax.text(x, -2.005, label, ha="center", va="bottom", fontsize=10, color=ink)
    for name, color, label in (
        ("sample_station", blue, "样品 / 返回站位"),
        ("stock_station", orange, "取货站位"),
    ):
        circle = stations[name]["circle"]
        xy, radius = circle["center_xy"], circle["radius_m"]
        x, y = xy
        ax.add_patch(Circle(xy, radius, color=color, alpha=0.17))
        ax.add_patch(Circle(xy, radius, fill=False, edgecolor=color, linewidth=2))
        ax.plot(x, y, "+", color=color, markersize=9, markeredgewidth=1.6)
        ax.text(x, -1.30, label, ha="center", color=color, weight="bold", fontsize=12)
        ax.text(x, -1.42, f"中心 ({x:.2f}, {y:.2f}) m", ha="center", fontsize=10, color=muted)
        ax.add_patch(
            FancyArrowPatch(
                (x, y - 0.075),
                (x, y - 0.27),
                arrowstyle="-|>",
                mutation_scale=17,
                linewidth=2,
                color=color,
            )
        )
    ax.text(2.12, -1.96, "共同朝向 −90°（朝向岛台）", ha="center", color=muted, fontsize=10)
    ax.annotate(
        "",
        xy=(3.22, -1.565),
        xytext=(1.08, -1.565),
        arrowprops={"arrowstyle": "->", "color": orange, "lw": 2},
    )
    ax.annotate(
        "",
        xy=(1.08, -1.695),
        xytext=(3.22, -1.695),
        arrowprops={"arrowstyle": "->", "color": blue, "lw": 2},
    )
    ax.text(
        2.15, -1.535, "向右平移取货 · 两站中心相距 2.30 m", ha="center", color=orange, fontsize=10
    )
    ax.text(2.15, -1.80, "向左平移送回 · 复用本次样品站位点", ha="center", color=blue, fontsize=10)
    ax.set_xlabel("世界坐标 X（m）", color=muted)
    ax.set_ylabel("世界坐标 Y（m）", color=muted)
    ax.set_title(
        "俯视位置关系 · 坐标和圆盘按同一比例绘制", loc="left", pad=14, fontsize=12, color=ink
    )
    ax.grid(alpha=0.12)
    for spine in ax.spines.values():
        spine.set_color("#DCE3E8")
    ax.tick_params(colors=muted, labelsize=9)

    for column, (name, color, label) in enumerate(
        (
            ("sample_station", blue, "样品 / 返回站位：放大"),
            ("stock_station", orange, "取货站位：放大"),
            ("sample_carton", green, "物体位置采样：尺度对照"),
        )
    ):
        zoom = fig.add_subplot(grid[1, column])
        if name == "sample_carton":
            points = np.array([plan.offset_xy_by_item[name] for plan in plans]) * 100
            radius_cm = object_radius * 100
        else:
            circle = stations[name]["circle"]
            points = (
                np.array([plan.base_xy_by_station[name] for plan in plans]) - circle["center_xy"]
            ) * 100
            radius_cm = circle["radius_m"] * 100
        zoom.add_patch(Circle((0, 0), radius_cm, facecolor=color, alpha=0.08))
        zoom.add_patch(Circle((0, 0), radius_cm, fill=False, edgecolor=color, linewidth=2))
        zoom.scatter(points[:, 0], points[:, 1], s=9, color=color, alpha=0.40, zorder=3)
        zoom.plot(0, 0, "+", color=ink, markersize=11, markeredgewidth=1.8, zorder=4)
        zoom.plot(
            *points[0],
            "*",
            color="#DB3159",
            markersize=13,
            markeredgecolor="white",
            markeredgewidth=0.7,
            zorder=5,
        )
        zoom.annotate(
            "",
            xy=(radius_cm, 0),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "<->", "color": ink, "lw": 1.2},
            zorder=6,
        )
        zoom.text(
            radius_cm / 2,
            0.45,
            f"R = {radius_cm:g} cm",
            ha="center",
            fontsize=10,
            color=ink,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2},
            zorder=7,
        )
        zoom.set(xlim=(-6.3, 6.3), ylim=(-6.3, 6.3), aspect="equal", xlabel="相对中心 ΔX（cm）")
        if column == 0:
            zoom.set_ylabel("相对中心 ΔY（cm）")
        zoom.set_title(label, loc="left", fontsize=12, color=color, pad=12)
        zoom.set_xticks([-5, 0, 5])
        zoom.set_yticks([-5, 0, 5])
        zoom.grid(alpha=0.15)
        zoom.tick_params(colors=muted, labelsize=9)
        for spine in zoom.spines.values():
            spine.set_color("#DCE3E8")
    handles = [
        Patch(facecolor="#DEEBF4", edgecolor=blue, label="圆内面积均匀采样"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            color=muted,
            markersize=4,
            label="真实解析点：seeds 0–199",
        ),
        Line2D(
            [],
            [],
            marker="*",
            linestyle="none",
            color="#DB3159",
            markersize=10,
            label="seed 0 的一次抽样",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.055),
        ncol=3,
        frameon=False,
        fontsize=10,
    )
    fig.text(
        0.5,
        0.025,
        "每个 episode 两站各抽一个点；初始、观察与返回共用样品点。圆表示基座目标采样范围，实际到达允许 8 cm 停止误差。",
        ha="center",
        fontsize=10,
        color=muted,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180, facecolor="white")
    plt.close(fig)
    print(args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
