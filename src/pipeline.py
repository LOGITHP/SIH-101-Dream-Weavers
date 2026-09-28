"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: End-to-End Master Pipeline Coordinator

Coordinates the full autonomous cycle:
1. Official Profile Ingestion
2. Machine Learning Competency Assessment (Random Forest)
3. Skill Gap Detection & Classification (Against Role Benchmarks)
4. Gap Priority Ranking with Transparent Explanations
5. Personalized Training Recommendation & Multi-phase Learning Path (iGOT / NSSTA)
6. Dynamic AI Quiz Generation on Target Gaps
7. Quiz Grading & Competency Score Update
8. Analytics & Visualization Generation (Radar, Gap Bar, Improvement Charts)
"""

import os
import sys
from typing import Dict, Any

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.competency_model import CompetencyPredictor
from src.skill_gap import SkillGapEngine, SKILL_NAMES
from src.recommendation import CourseRecommendationEngine
from src.quiz_generator import AIQuizGenerator
from src.visualizer import OfficialAnalyticsVisualizer


class PersonalizedLearningPipeline:
    """
    Unified pipeline powering the AI learning platform for India's Official Statistical System.
    """

    def __init__(
        self,
        model_path="models/competency_model.pkl",
        preprocessor_path="models/preprocessing.pkl",
        catalogue_path="data/training_catalogue.csv"
    ):
        self.predictor = CompetencyPredictor(model_path=model_path, preprocessor_path=preprocessor_path)
        self.gap_engine = SkillGapEngine()
        self.rec_engine = CourseRecommendationEngine(catalogue_path=catalogue_path)
        self.quiz_gen = AIQuizGenerator()
        self.visualizer = OfficialAnalyticsVisualizer()

    def run_assessment_and_recommendation(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes stages 1 through 5 of the pipeline.
        """
        target_role = profile.get("target_role", profile.get("role", "Statistical Analyst"))

        # 1. Competency Prediction via ML
        competency_result = self.predictor.predict(profile)
        pred_competency = competency_result["predicted_competency"]

        # 2. Skill Gap Analysis
        skill_scores = {k: int(profile.get(k, 0)) for k in SKILL_NAMES}
        gap_analysis = self.gap_engine.analyze_gaps(skill_scores, target_role)

        # 3. Personalized Course Recommendation & Learning Path
        rec_result = self.rec_engine.generate_learning_path(
            official_profile=profile,
            skill_gap_analysis=gap_analysis,
            competency_level=pred_competency
        )

        # 4. Generate Visualizations (Radar & Bar charts)
        benchmark_reqs = self.gap_engine.get_role_benchmark(target_role)
        emp_safe_id = profile.get("employee_id", "demo").replace("-", "_")

        radar_b64 = self.visualizer.plot_skill_radar_chart(
            current_scores=skill_scores,
            required_scores=benchmark_reqs,
            role_name=target_role,
            save_name=f"radar_{emp_safe_id}.png"
        )

        gap_bar_b64 = self.visualizer.plot_skill_gap_bar_chart(
            ranked_gaps=gap_analysis["ranked_gaps"],
            save_name=f"gap_bar_{emp_safe_id}.png"
        )

        # 5. Pre-generate targeted quiz on the highest priority gap
        primary_gap = gap_analysis["active_priority_gaps"][0] if gap_analysis["active_priority_gaps"] else None
        target_skill = primary_gap["skill_key"] if primary_gap else "statistical_analysis_score"
        quiz_questions = self.quiz_gen.generate_quiz_from_material(
            learning_material=f"Indian Official Statistics training for {target_skill}",
            num_questions=5,
            target_skill=target_skill,
            difficulty=pred_competency
        )

        return {
            "official_id": profile.get("employee_id"),
            "name": profile.get("name", "Official"),
            "role": profile.get("role"),
            "target_role": target_role,
            "department": profile.get("department"),
            "experience_years": profile.get("experience_years"),
            "competency_assessment": competency_result,
            "skill_scores": skill_scores,
            "benchmark_requirements": benchmark_reqs,
            "skill_gap_analysis": gap_analysis,
            "recommended_courses": rec_result["recommended_courses"],
            "learning_path": rec_result["learning_path"],
            "diagnostic_quiz": {
                "targeted_skill": target_skill,
                "targeted_skill_display": primary_gap["skill_name"] if primary_gap else "Statistical Analysis",
                "questions": quiz_questions
            },
            "visualizations": {
                "radar_chart": radar_b64,
                "gap_bar_chart": gap_bar_b64
            }
        }

    def process_quiz_and_update_competency(
        self,
        profile: Dict[str, Any],
        quiz_questions: list,
        submitted_answers: dict
    ) -> Dict[str, Any]:
        """
        Executes stages 6 through 8:
        Evaluates submitted quiz, calculates skill improvement, and updates competency level.
        """
        evaluation = self.quiz_gen.evaluate_quiz_submission(quiz_questions, submitted_answers)

        boost = evaluation["estimated_skill_boost"]
        updated_profile = dict(profile)

        # Find target skill addressed
        target_skills = set(q.get("related_skill", "statistical_analysis_score") for q in quiz_questions)
        before_scores = {k: int(profile.get(k, 0)) for k in SKILL_NAMES}
        after_scores = dict(before_scores)

        for sk in target_skills:
            if sk in after_scores:
                after_scores[sk] = min(99, after_scores[sk] + boost)
                updated_profile[sk] = after_scores[sk]

        # Update assessment score and training stats
        updated_profile["assessment_score"] = min(100, int(profile.get("assessment_score", 60)) + int(boost * 0.8))
        updated_profile["training_hours_completed"] = int(profile.get("training_hours_completed", 0)) + 15
        updated_profile["previous_training_count"] = int(profile.get("previous_training_count", 0)) + 1
        updated_profile["learning_progress"] = min(100, int(profile.get("learning_progress", 0)) + 20)

        # Re-predict competency with improved scores
        new_competency = self.predictor.predict(updated_profile)

        # Re-calculate remaining gaps
        target_role = updated_profile.get("target_role", updated_profile.get("role", "Statistical Analyst"))
        new_gaps = self.gap_engine.analyze_gaps(after_scores, target_role)

        # Generate Before/After chart
        emp_safe_id = profile.get("employee_id", "demo").replace("-", "_")
        improvement_chart_b64 = self.visualizer.plot_competency_improvement(
            before_scores=before_scores,
            after_scores=after_scores,
            save_name=f"improvement_{emp_safe_id}.png"
        )

        return {
            "quiz_evaluation": evaluation,
            "skill_points_gained": boost,
            "skills_updated": list(target_skills),
            "before_scores": before_scores,
            "after_scores": after_scores,
            "previous_competency": profile.get("competency_level", "Developing"),
            "updated_competency": new_competency["predicted_competency"],
            "updated_confidence": new_competency["confidence"],
            "remaining_gaps_count": new_gaps["skills_with_gaps"],
            "improvement_chart": improvement_chart_b64,
            "updated_profile": updated_profile
        }


if __name__ == "__main__":
    from src.demo_profiles import DEMO_PROFILES
    pipeline = PersonalizedLearningPipeline()
    res = pipeline.run_assessment_and_recommendation(DEMO_PROFILES["employee_a"])
    print(f"Pipeline executed successfully for {res['name']} ({res['role']})")
    print(f"Predicted Competency: {res['competency_assessment']['predicted_competency']} (Confidence: {res['competency_assessment']['confidence'] * 100:.1f}%)")
    print(f"Active Gaps Count: {len(res['skill_gap_analysis']['active_priority_gaps'])}")
    print(f"Recommended Courses: {len(res['recommended_courses'])}")
