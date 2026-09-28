"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Verified Demo Profiles for India's Official Statistical System

Provides 3 canonical employee profiles representing diverse roles, departments,
experience levels, and skill deficit patterns across MoSPI, CSO, NSSO, and DQAD.
"""

from typing import Dict, Any

DEMO_PROFILES: Dict[str, Dict[str, Any]] = {
    "employee_a": {
        "employee_id": "IND-OSS-0101",
        "name": "Dr. Rajesh Sharma",
        "role": "Statistical Analyst",
        "department": "Central Statistics Office (CSO)",
        "target_role": "Senior Statistical Officer (SSO)",
        "experience_years": 4,
        "statistical_analysis_score": 86,
        "data_visualization_score": 68,
        "python_score": 42,
        "sql_score": 52,
        "data_collection_score": 65,
        "data_quality_score": 82,
        "statistical_methods_score": 84,
        "communication_score": 74,
        "previous_training_count": 3,
        "training_hours_completed": 60,
        "assessment_score": 72,
        "learning_progress": 45,
        "bio": "Economics doctorate with strong classical statistical theory. Needs automated computing skills (Python, SQL) to transition into modern large-scale survey processing."
    },
    "employee_b": {
        "employee_id": "IND-OSS-0202",
        "name": "Pooja Verma",
        "role": "Data Analyst",
        "department": "Data Quality Assurance Division (DQAD)",
        "target_role": "Statistical Analyst",
        "experience_years": 2,
        "statistical_analysis_score": 64,
        "data_visualization_score": 86,
        "python_score": 88,
        "sql_score": 84,
        "data_collection_score": 48,
        "data_quality_score": 78,
        "statistical_methods_score": 54,
        "communication_score": 76,
        "previous_training_count": 2,
        "training_hours_completed": 40,
        "assessment_score": 68,
        "learning_progress": 30,
        "bio": "Computer science background with exceptional programming and database skills. Requires grounding in official survey methodologies and stratified sampling theory."
    },
    "employee_c": {
        "employee_id": "IND-OSS-0303",
        "name": "Amitabh Sen",
        "role": "Junior Statistical Officer (JSO)",
        "department": "National Sample Survey Office (NSSO)",
        "target_role": "Assistant Director (Statistics)",
        "experience_years": 8,
        "statistical_analysis_score": 68,
        "data_visualization_score": 55,
        "python_score": 38,
        "sql_score": 50,
        "data_collection_score": 92,
        "data_quality_score": 86,
        "statistical_methods_score": 76,
        "communication_score": 82,
        "previous_training_count": 6,
        "training_hours_completed": 120,
        "assessment_score": 74,
        "learning_progress": 60,
        "bio": "Veteran field coordinator with exceptional CAPI and grassroots data collection mastery. Targeting leadership role as Assistant Director requiring econometric modeling & visualization."
    }
}


def get_demo_profile(key: str) -> Dict[str, Any]:
    """Retrieves demo profile by key ('employee_a', 'employee_b', 'employee_c')."""
    return DEMO_PROFILES.get(key, DEMO_PROFILES["employee_a"])


def list_demo_profiles() -> list:
    """Returns overview list of demo profiles."""
    return [
        {
            "key": k,
            "employee_id": v["employee_id"],
            "name": v["name"],
            "role": v["role"],
            "target_role": v["target_role"],
            "department": v["department"],
            "experience_years": v["experience_years"]
        }
        for k, v in DEMO_PROFILES.items()
    ]
