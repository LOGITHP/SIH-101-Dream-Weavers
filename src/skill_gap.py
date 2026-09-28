"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Skill Gap Detection & Priority Ranking Engine

Compares official's current skill levels against target role requirements:
- Computes exact numerical gap: Required - Current
- Classifies into tiers: No Gap, Low Gap, Medium Gap, High Gap
- Ranks gaps by urgency / priority
- Generates detailed, explainable rationales for each skill gap
"""

from typing import Dict, List, Any
from src.data_generation import ROLE_SKILL_REQUIREMENTS, SKILL_NAMES

SKILL_DISPLAY_NAMES = {
    "statistical_analysis_score": "Statistical Analysis & Inference",
    "data_visualization_score": "Data Visualization & Dashboarding",
    "python_score": "Python for Statistical Computing",
    "sql_score": "SQL & Relational Data Management",
    "data_collection_score": "Survey Data Collection & CAPI",
    "data_quality_score": "Data Quality Scrutiny & Validation",
    "statistical_methods_score": "Sampling & Statistical Methods",
    "communication_score": "Statistical Report Writing & Communication"
}


class SkillGapEngine:
    """
    Analyzes an official's competencies against role benchmark profiles.
    """

    def __init__(self, role_requirements: Dict[str, Dict[str, int]] = None):
        self.role_requirements = role_requirements or ROLE_SKILL_REQUIREMENTS

    def get_role_benchmark(self, role: str) -> Dict[str, int]:
        """
        Retrieves benchmark requirements for a specified role.
        Falls back to Statistical Analyst if role not found.
        """
        if role in self.role_requirements:
            return self.role_requirements[role]
        return self.role_requirements["Statistical Analyst"]

    def classify_gap(self, gap_value: int) -> Dict[str, Any]:
        """
        Categorizes numerical gap into standardized SIH tiers with priority weighting.
        """
        if gap_value <= 0:
            return {
                "category": "No Gap",
                "priority": "None",
                "priority_rank": 4,
                "urgency_badge": "Success",
                "action_recommended": "Maintain proficiency or mentor others."
            }
        elif gap_value <= 15:
            return {
                "category": "Low Gap",
                "priority": "Low",
                "priority_rank": 3,
                "urgency_badge": "Info",
                "action_recommended": "Short refresher modules or self-paced reading."
            }
        elif gap_value <= 30:
            return {
                "category": "Medium Gap",
                "priority": "Medium",
                "priority_rank": 2,
                "urgency_badge": "Warning",
                "action_recommended": "Structured medium-duration training course and hands-on lab."
            }
        else:
            return {
                "category": "High Gap",
                "priority": "High",
                "priority_rank": 1,
                "urgency_badge": "Critical",
                "action_recommended": "Mandatory intensive training with mentorship & assessment."
            }

    def generate_explanation(self, skill_key: str, current_score: int, required_score: int, gap: int, category: str, role: str) -> str:
        """
        Model Explainability: Generates human-readable, transparent reasoning.
        """
        display_name = SKILL_DISPLAY_NAMES.get(skill_key, skill_key.replace("_score", "").replace("_", " ").title())

        if gap <= 0:
            surplus = abs(gap)
            return (
                f"{display_name} has NO competency gap. The official's score ({current_score}/100) exceeds "
                f"or meets the benchmark ({required_score}/100) for '{role}' by {surplus} point(s)."
            )
        elif category == "Low Gap":
            return (
                f"{display_name} was classified as a Low Priority Gap. Current score is {current_score}/100 versus "
                f"required {required_score}/100 for '{role}' (deficit of {gap} points). Minor targeted practice needed."
            )
        elif category == "Medium Gap":
            return (
                f"{display_name} was identified as a Medium Priority Gap. Current score of {current_score}/100 falls "
                f"moderately short of the required {required_score}/100 (deficit of {gap} points). Formal training recommended."
            )
        else:
            return (
                f"{display_name} was identified as a HIGH PRIORITY CRITICAL GAP because current score is {current_score}/100 "
                f"while the required competency level for '{role}' is {required_score}/100 (critical deficit of {gap} points). "
                f"Immediate capability enhancement required."
            )

    def analyze_gaps(self, official_scores: Dict[str, int], target_role: str) -> Dict[str, Any]:
        """
        Main execution method:
        Takes official's current skill scores and target role, computes gaps,
        sorts by priority, and returns structured analysis with explainability.
        """
        benchmark = self.get_role_benchmark(target_role)
        gap_items = []

        total_gap = 0
        gap_count = 0

        for skill_key, required_score in benchmark.items():
            current_score = int(official_scores.get(skill_key, 0))
            gap = required_score - current_score
            classification = self.classify_gap(gap)
            explanation = self.generate_explanation(
                skill_key=skill_key,
                current_score=current_score,
                required_score=required_score,
                gap=gap,
                category=classification["category"],
                role=target_role
            )

            if gap > 0:
                total_gap += gap
                gap_count += 1

            gap_items.append({
                "skill_key": skill_key,
                "skill_name": SKILL_DISPLAY_NAMES.get(skill_key, skill_key),
                "current_score": current_score,
                "required_score": required_score,
                "gap": gap,
                "category": classification["category"],
                "priority": classification["priority"],
                "priority_rank": classification["priority_rank"],
                "urgency_badge": classification["urgency_badge"],
                "action_recommended": classification["action_recommended"],
                "explanation": explanation
            })

        # Sort gaps: High priority (rank 1) and largest gap first
        sorted_gaps = sorted(gap_items, key=lambda x: (x["priority_rank"], -x["gap"]))

        # Filter only active gaps for the priority ranking list
        active_priority_gaps = [g for g in sorted_gaps if g["gap"] > 0]

        summary = {
            "target_role": target_role,
            "total_skills_evaluated": len(gap_items),
            "skills_with_gaps": gap_count,
            "skills_meeting_benchmark": len(gap_items) - gap_count,
            "average_gap_magnitude": round(total_gap / max(1, gap_count), 2),
            "priority_breakdown": {
                "high": sum(1 for g in gap_items if g["category"] == "High Gap"),
                "medium": sum(1 for g in gap_items if g["category"] == "Medium Gap"),
                "low": sum(1 for g in gap_items if g["category"] == "Low Gap"),
                "no_gap": sum(1 for g in gap_items if g["category"] == "No Gap")
            },
            "ranked_gaps": sorted_gaps,
            "active_priority_gaps": active_priority_gaps
        }
        return summary


if __name__ == "__main__":
    engine = SkillGapEngine()
    test_scores = {
        "python_score": 45,
        "sql_score": 60,
        "statistical_analysis_score": 85,
        "data_visualization_score": 70,
        "data_collection_score": 75,
        "data_quality_score": 80,
        "statistical_methods_score": 78,
        "communication_score": 65
    }
    analysis = engine.analyze_gaps(test_scores, "Statistical Analyst")
    print("Gap Analysis Summary:")
    print("Active Priority Gaps:")
    for g in analysis["active_priority_gaps"]:
        print(f"  * {g['skill_name']}: Current={g['current_score']}, Req={g['required_score']}, Gap={g['gap']} [{g['priority']} Priority]")
        print(f"    Rationale: {g['explanation']}")
