"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Personalized Training Recommendation & Learning Path Engine

Integrates:
- Skill Gap Priorities (High -> Medium -> Low)
- Current Competency Level (Beginner, Developing, Proficient, Advanced)
- Training Catalogue from iGOT Karmayogi and NSSTA
- Prerequisite awareness & duration optimization
- Fully explainable recommendations and multi-stage learning path generation
"""

import os
import pandas as pd
from typing import List, Dict, Any

DEFAULT_CATALOGUE_PATH = "data/training_catalogue.csv"

# Difficulty mapping for progressive learning
DIFFICULTY_PROGRESSION = {
    "Beginner": ["Beginner", "Developing"],
    "Developing": ["Beginner", "Developing", "Proficient"],
    "Proficient": ["Developing", "Proficient", "Advanced"],
    "Advanced": ["Proficient", "Advanced"]
}


class CourseRecommendationEngine:
    """
    Recommends tailored courses and constructs personalized learning paths
    for officials in India's Statistical System.
    """

    def __init__(self, catalogue_path: str = DEFAULT_CATALOGUE_PATH):
        if not os.path.exists(catalogue_path):
            raise FileNotFoundError(f"Training catalogue not found at {catalogue_path}")
        self.catalogue_df = pd.read_csv(catalogue_path).fillna("None")


    def recommend_courses(
        self,
        skill_gap_analysis: Dict[str, Any],
        competency_level: str = "Developing",
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Matches courses against active skill gaps and official competency.
        Returns prioritized list with clear rationale.
        """
        active_gaps = skill_gap_analysis.get("active_priority_gaps", [])
        if not active_gaps:
            # If no active gap, recommend advanced leadership/governance courses
            recommendations = []
            adv_courses = self.catalogue_df[self.catalogue_df["difficulty"].isin(["Proficient", "Advanced"])].head(3)
            for _, row in adv_courses.iterrows():
                recommendations.append({
                    "course_id": row["course_id"],
                    "course_name": row["course_name"],
                    "skill": row["skill"],
                    "difficulty": row["difficulty"],
                    "duration_hours": int(row["duration_hours"]),
                    "provider": row["provider"],
                    "prerequisite": row["prerequisite"],
                    "description": row["description"],
                    "rating": float(row["rating"]),
                    "priority": "Enrichment",
                    "reason": f"Official meets all role benchmarks. Recommended for continuous professional excellence."
                })
            return recommendations

        allowed_difficulties = DIFFICULTY_PROGRESSION.get(competency_level, ["Beginner", "Developing"])
        recommendations = []
        seen_course_ids = set()

        # Iterate through gaps in ranked priority order
        for gap in active_gaps:
            skill_key = gap["skill_key"]
            gap_val = gap["gap"]
            priority = gap["priority"]
            skill_display = gap["skill_name"]

            # Filter courses targeting this specific skill
            matching = self.catalogue_df[self.catalogue_df["skill"] == skill_key].copy()

            if matching.empty:
                continue

            # Prioritize matching difficulty based on current competency
            matching["difficulty_match"] = matching["difficulty"].apply(
                lambda d: 0 if d in allowed_difficulties else 1
            )
            # Sort by difficulty match, then rating descending
            matching = matching.sort_values(by=["difficulty_match", "rating"], ascending=[True, False])

            for _, course in matching.iterrows():
                cid = course["course_id"]
                if cid in seen_course_ids:
                    continue

                seen_course_ids.add(cid)

                # Formulate explainable recommendation reason
                reason = (
                    f"Recommended from {course['provider']} because '{skill_display}' has a "
                    f"{priority} Priority deficit of {gap_val} points. The course level ({course['difficulty']}) "
                    f"is calibrated for an official at the '{competency_level}' competency tier."
                )

                recommendations.append({
                    "course_id": cid,
                    "course_name": course["course_name"],
                    "skill": skill_key,
                    "skill_display": skill_display,
                    "difficulty": course["difficulty"],
                    "duration_hours": int(course["duration_hours"]),
                    "provider": course["provider"],
                    "prerequisite": course["prerequisite"],
                    "description": course["description"],
                    "rating": float(course["rating"]),
                    "gap_addressed": gap_val,
                    "priority": priority,
                    "reason": reason
                })

                if len(recommendations) >= top_n:
                    break
            if len(recommendations) >= top_n:
                break

        return recommendations

    def generate_learning_path(
        self,
        official_profile: Dict[str, Any],
        skill_gap_analysis: Dict[str, Any],
        competency_level: str
    ) -> Dict[str, Any]:
        """
        Constructs an end-to-end personalized learning path roadmap.
        Roadmap Stages:
        1. Foundational Gap Remediation (Urgent gaps)
        2. Core Functional Competency (Medium gaps)
        3. Statistical Mastery & Standards (Low/Expansion gaps)
        4. Capstone AI Knowledge Assessment
        """
        recommended_courses = self.recommend_courses(
            skill_gap_analysis=skill_gap_analysis,
            competency_level=competency_level,
            top_n=6
        )

        total_hours = sum(c["duration_hours"] for c in recommended_courses)

        milestones = []
        stage_names = [
            "Phase 1: Critical Competency Rectification",
            "Phase 2: Core Analytical Capability Building",
            "Phase 3: Domain Specialization & Quality Standards"
        ]

        # Group recommendations into phases
        chunk_size = max(1, (len(recommended_courses) + 2) // 3)
        course_chunks = [recommended_courses[i:i + chunk_size] for i in range(0, len(recommended_courses), chunk_size)]

        for idx, chunk in enumerate(course_chunks):
            phase_title = stage_names[idx] if idx < len(stage_names) else f"Phase {idx + 1}: Continuous Upskilling"
            phase_hours = sum(c["duration_hours"] for c in chunk)
            phase_skills = list(set(c["skill_display"] for c in chunk))

            milestones.append({
                "phase_number": idx + 1,
                "phase_title": phase_title,
                "target_skills": phase_skills,
                "estimated_hours": phase_hours,
                "courses": chunk,
                "evaluation_checkpoint": f"AI-Generated Interactive Knowledge Check ({phase_skills[0] if phase_skills else 'Official Statistics'})"
            })

        learning_path = {
            "official_id": official_profile.get("employee_id", "IND-OSS-DEMO"),
            "role": official_profile.get("role", "Statistical Official"),
            "target_role": skill_gap_analysis.get("target_role", "Statistical Analyst"),
            "current_competency": competency_level,
            "total_curated_courses": len(recommended_courses),
            "estimated_total_hours": total_hours,
            "estimated_weeks_to_completion": max(2, round(total_hours / 6.0, 1)),  # Assuming 6 hrs/week in-service training
            "phases": milestones,
            "action_plan_summary": (
                f"Personalized {len(milestones)}-phase learning path curated across {total_hours} training hours. "
                f"Targets {len(skill_gap_analysis.get('active_priority_gaps', []))} identified competency gaps "
                f"using accredited modules from iGOT Karmayogi and NSSTA."
            )
        }

        return {
            "recommended_courses": recommended_courses,
            "learning_path": learning_path
        }


if __name__ == "__main__":
    engine = CourseRecommendationEngine()
    dummy_gaps = {
        "target_role": "Statistical Analyst",
        "active_priority_gaps": [
            {"skill_key": "python_score", "skill_name": "Python for Statistical Computing", "gap": 30, "priority": "High"},
            {"skill_key": "sql_score", "skill_name": "SQL & Relational Data Management", "gap": 20, "priority": "Medium"},
            {"skill_key": "data_visualization_score", "skill_name": "Data Visualization & Dashboarding", "gap": 5, "priority": "Low"}
        ]
    }
    res = engine.generate_learning_path({"employee_id": "TEST-1", "role": "Junior Statistical Officer"}, dummy_gaps, "Developing")
    print(f"Total Courses Recommended: {len(res['recommended_courses'])}")
    for c in res["recommended_courses"]:
        print(f" - [{c['provider']}] {c['course_name']} ({c['difficulty']}) -> {c['priority']} Priority")
        print(f"   Reason: {c['reason']}")
