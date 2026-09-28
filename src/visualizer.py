"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Visualizations & Analytics Engine

Generates:
1. Skill Radar Chart (Current vs Required benchmarks)
2. Current vs Required Skill Comparison Bar Chart
3. Skill Gap Deficit Bar Chart with Priority Color Coding
4. Competency Distribution in Statistical System
5. Before & After Competency Improvement Progress Chart
"""

import os
import io
import base64
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.skill_gap import SKILL_DISPLAY_NAMES


class OfficialAnalyticsVisualizer:
    """
    Visualizer producing professional analytical charts for official dashboards.
    """

    def __init__(self, output_dir: str = "models/plots"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        # Configure professional aesthetic
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    def _fig_to_base64(self, fig) -> str:
        """Converts matplotlib figure to base64 string for direct web/API embedding."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        plt.close(fig)
        return f"data:image/png;base64,{img_b64}"

    def plot_skill_radar_chart(
        self,
        current_scores: dict,
        required_scores: dict,
        role_name: str = "Target Role",
        save_name: str = "skill_radar_chart.png"
    ) -> str:
        """
        Creates a multi-axis spider / radar chart comparing official's current skills
        against target role requirements.
        """
        categories = list(required_scores.keys())
        display_labels = [SKILL_DISPLAY_NAMES.get(k, k).split()[0] for k in categories]

        num_vars = len(categories)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()

        # Complete the loop
        angles += angles[:1]
        cur_vals = [current_scores.get(k, 0) for k in categories] + [current_scores.get(categories[0], 0)]
        req_vals = [required_scores.get(k, 0) for k in categories] + [required_scores.get(categories[0], 0)]

        fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

        # Required benchmark line
        ax.plot(angles, req_vals, color="#e65100", linewidth=2.5, linestyle="--", label=f"Required ({role_name})")
        ax.fill(angles, req_vals, color="#ff9800", alpha=0.15)

        # Current skill line
        ax.plot(angles, cur_vals, color="#1565c0", linewidth=2.5, label="Current Competency")
        ax.fill(angles, cur_vals, color="#2196f3", alpha=0.35)

        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.degrees(angles[:-1]), display_labels, fontsize=10, fontweight="bold")

        ax.set_rlim(0, 100)
        ax.set_rticks([20, 40, 60, 80, 100])
        ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="#555")
        ax.set_title(f"Competency Radar: Current vs Required\n({role_name})", fontsize=12, fontweight="bold", pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")

        b64 = self._fig_to_base64(fig)
        return b64

    def plot_skill_gap_bar_chart(
        self,
        ranked_gaps: list,
        save_name: str = "skill_gap_bar_chart.png"
    ) -> str:
        """
        Creates a color-coded bar chart illustrating each competency gap magnitude and priority.
        """
        skills = [g["skill_name"] for g in ranked_gaps]
        gaps = [g["gap"] for g in ranked_gaps]
        priorities = [g["priority"] for g in ranked_gaps]

        # Color mapping by priority
        color_map = {
            "High": "#d32f2f",     # Red
            "Medium": "#f57c00",   # Amber / Orange
            "Low": "#fbc02d",      # Yellow
            "None": "#388e3c"      # Green (Surplus/No gap)
        }
        colors = [color_map.get(p, "#90caf9") for p in priorities]

        fig, ax = plt.subplots(figsize=(9, 5))
        bars = ax.barh(skills[::-1], gaps[::-1], color=colors[::-1], height=0.6, edgecolor="#333", alpha=0.9)

        ax.axvline(0, color="black", linewidth=1)
        ax.axvline(15, color="#fbc02d", linestyle=":", alpha=0.7, label="Low Threshold (15)")
        ax.axvline(30, color="#d32f2f", linestyle=":", alpha=0.7, label="High Threshold (30)")

        # Value annotations
        for bar in bars:
            width = bar.get_width()
            pos_x = width + 1 if width >= 0 else width - 4
            ax.annotate(f"{width:+d}",
                        xy=(pos_x, bar.get_y() + bar.get_height() / 2),
                        xytext=(0, 0), textcoords="offset points",
                        ha="left" if width >= 0 else "right", va="center",
                        fontsize=9, fontweight="bold")

        ax.set_xlabel("Skill Deficit / Gap Magnitude (Required - Current)", fontsize=11, fontweight="bold")
        ax.set_title("Skill Gap Deficit & Priority Breakdown\n(Smart India Hackathon 2026 - Dream Weavers)", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()

        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")

        b64 = self._fig_to_base64(fig)
        return b64

    def plot_competency_improvement(
        self,
        before_scores: dict,
        after_scores: dict,
        save_name: str = "competency_improvement.png"
    ) -> str:
        """
        Visualizes before vs after competency progression after training/quiz completion.
        """
        categories = list(before_scores.keys())
        labels = [SKILL_DISPLAY_NAMES.get(k, k).split()[0] for k in categories]

        x = np.arange(len(categories))
        width = 0.35

        before_vals = [before_scores[k] for k in categories]
        after_vals = [after_scores[k] for k in categories]

        fig, ax = plt.subplots(figsize=(9, 5))
        rects1 = ax.bar(x - width / 2, before_vals, width, label="Baseline Pre-Training", color="#78909c", alpha=0.85)
        rects2 = ax.bar(x + width / 2, after_vals, width, label="Post-Training / Quiz Evaluation", color="#2e7d32", alpha=0.9)

        for rect in rects2:
            h = rect.get_height()
            ax.annotate(f"{h}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

        ax.set_ylabel("Proficiency Score (0 - 100)", fontsize=11, fontweight="bold")
        ax.set_title("Official Competency Progression Before vs After Personalized Learning", fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylim(0, 115)
        ax.legend(loc="upper left")
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        fig.tight_layout()

        filepath = os.path.join(self.output_dir, save_name)
        fig.savefig(filepath, dpi=200, bbox_inches="tight")

        b64 = self._fig_to_base64(fig)
        return b64


if __name__ == "__main__":
    vis = OfficialAnalyticsVisualizer()
    cur = {
        "statistical_analysis_score": 85,
        "data_visualization_score": 60,
        "python_score": 45,
        "sql_score": 50,
        "data_collection_score": 75,
        "data_quality_score": 80,
        "statistical_methods_score": 75,
        "communication_score": 70
    }
    req = {
        "statistical_analysis_score": 85,
        "data_visualization_score": 75,
        "python_score": 75,
        "sql_score": 80,
        "data_collection_score": 60,
        "data_quality_score": 80,
        "statistical_methods_score": 80,
        "communication_score": 70
    }
    vis.plot_skill_radar_chart(cur, req, "Statistical Analyst")
    print("Visualizer tested successfully.")
