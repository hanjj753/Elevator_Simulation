"""Render a research figure from the saved controlled experiment.

Simulation and statistical analysis have no third-party dependencies.
This optional plotting step requires matplotlib.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("output/ablation/results.json"))
    args = parser.parse_args()
    report = json.loads(args.input.read_text(encoding="utf-8"))
    font_path = Path("C:/Windows/Fonts/malgun.ttf")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update({"font.size": 11, "axes.unicode_minus": False,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    patterns = {"up_peak": "출근형", "interfloor": "층간형", "down_peak": "퇴근형"}
    levels = {"normal": "보통", "congested": "혼잡"}
    rows = [row for row in report["contrasts"] if row["contrast"] == "age_order_added_effect"]
    labels = [f"{levels[row['traffic_level']]} · {patterns[row['traffic_pattern']]}" for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), sharey=True)
    panels = [
        ("p95_wait_seconds", "전체 승객의 P95", "#176A91"),
        ("eta_tail5_mean_wait_seconds", "기본 ETA에서 가장 오래 기다린 5%의 평균", "#A44A28"),
    ]
    for axis, (metric, title, color) in zip(axes, panels):
        for y, row in enumerate(rows):
            estimate = row["metrics"][metric]
            mean = estimate["mean_difference_seconds"]
            low, high = estimate["ci95_low_seconds"], estimate["ci95_high_seconds"]
            # Intervals can be asymmetric and need not contain the point estimate.
            axis.plot([low, high], [y, y], color=color, lw=2)
            axis.plot([low, low], [y-.08, y+.08], color=color, lw=1.5)
            axis.plot([high, high], [y-.08, y+.08], color=color, lw=1.5)
            axis.scatter([mean], [y], color=color, s=45, zorder=3)
        axis.axvline(0, color="#5C6166", lw=1, linestyle="--")
        axis.set_title(title, fontsize=12, pad=14)
        axis.set_xlabel("오래된 호출 우선 - 무작위 순서 재배분 (초)", labelpad=12)
        axis.grid(axis="x", color="#E3E6E8", linewidth=.7)
        axis.set_axisbelow(True)
    axes[0].set_yticks(range(len(rows)), labels)
    axes[0].invert_yaxis()
    fig.suptitle("재배분 조건을 맞췄을 때 오래된 호출 순서의 추가 효과", fontsize=16, y=.98)
    runs = report["protocol"]["demand_runs_per_condition"]
    orders = len(report["protocol"]["order_seeds"])
    fig.text(.5, .04,
             f"음수: 오래된 호출 우선에 유리. 점: 평균 차이, 선: 탐색적 95% 짝지은 부트스트랩 구간.\n"
             f"조건당 승객 발생 씨앗 {runs}개. 무작위 순서 {orders}회는 씨앗 안에서 먼저 평균. 다중비교 보정 없음.",
             ha="center", va="bottom", fontsize=10, color="#40464C")
    fig.subplots_adjust(left=.12, right=.98, top=.83, bottom=.23, wspace=.25)
    for extension in ("png", "svg"):
        path = args.input.parent / f"age_order_added_effect.{extension}"
        fig.savefig(path, dpi=240, facecolor="white")
        print(path.resolve())
    plt.close(fig)

    subgroup_path = args.input.parent / "subgroups.json"
    if subgroup_path.exists():
        subgroup = json.loads(subgroup_path.read_text(encoding="utf-8"))
        rows = [row for row in subgroup["effects"]
                if row["traffic_level"] == "congested" and row["traffic_pattern"] == "down_peak"]
        names = {"lobby": "1층", "lower": "2-9층", "middle": "10-17층", "upper": "18-25층"}
        fig, axis = plt.subplots(figsize=(9, 5.4))
        for y, row in enumerate(rows):
            estimate = row["effect"]
            low, high = estimate["ci95_low_seconds"], estimate["ci95_high_seconds"]
            axis.plot([low, high], [y, y], color="#176A91", lw=2)
            axis.scatter([estimate["mean_difference_seconds"]], [y], color="#176A91", s=50, zorder=3)
        axis.set_yticks(range(len(rows)), [
            f"{names[row['origin_floor_band']]} (승객 {row['passengers_across_demand_runs']:,}명)" for row in rows
        ])
        axis.invert_yaxis()
        axis.axvline(0, color="#5C6166", linestyle="--", lw=1)
        axis.grid(axis="x", color="#E3E6E8", linewidth=.7)
        axis.set_axisbelow(True)
        axis.set_xlabel("평균 대기시간 차이: 오래된 호출 우선 - 무작위 순서 재배분 (초)", labelpad=12)
        axis.set_title("혼잡 퇴근형: 나이 순서의 영향은 출발층에 따라 달랐다", fontsize=15, pad=18)
        fig.text(.5, .035,
                 "결과 확인 후 추가한 탐색적 층별 분석. 점: 평균 차이, 선: 보정하지 않은 95% 구간.\n"
                 "각 구간 안에서는 승객 수로 가중하고, 반복 간에는 동일 가중. 1층은 28회, 나머지는 30회 관측.",
                 ha="center", va="bottom", fontsize=10, color="#40464C")
        fig.subplots_adjust(left=.3, right=.97, top=.85, bottom=.25)
        for extension in ("png", "svg"):
            path = args.input.parent / f"congested_down_peak_floor_effect.{extension}"
            fig.savefig(path, dpi=240, facecolor="white")
            print(path.resolve())
        plt.close(fig)


if __name__ == "__main__":
    main()
