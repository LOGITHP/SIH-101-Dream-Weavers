"""
Smart India Hackathon 2026 - Problem Statement SIH26101
Team: Dream Weavers (AD14)
Module: Realistic Synthetic Data Generator for India's Official Statistical System

DISCLAIMER: This dataset contains SYNTHETIC / DEMO DATA generated for academic and prototype
testing purposes. It does NOT contain real personal data or official records from the Government
of India, MoSPI, NSSO, CSO, or any associated departments.
"""

import os
import json
import random
import numpy as np
import pandas as pd

# Define official roles in the Indian Statistical System
ROLES = [
    "Statistical Analyst",
    "Data Analyst",
    "Junior Statistical Officer (JSO)",
    "Senior Statistical Officer (SSO)",
    "Field Operations Officer (NSSO)",
    "Assistant Director (Statistics)",
    "Survey Data Specialist"
]

DEPARTMENTS = [
    "National Sample Survey Office (NSSO)",
    "Central Statistics Office (CSO)",
    "National Statistical Systems Training Academy (NSSTA)",
    "Economic Statistics Division (ESD)",
    "Social Statistics Division (SSD)",
    "Data Quality Assurance Division (DQAD)",
    "State Directorate of Economics and Statistics (DES)"
]

SKILL_NAMES = [
    "statistical_analysis_score",
    "data_visualization_score",
    "python_score",
    "sql_score",
    "data_collection_score",
    "data_quality_score",
    "statistical_methods_score",
    "communication_score"
]

# Benchmark required skill levels (0-100) per role
ROLE_SKILL_REQUIREMENTS = {
    "Statistical Analyst": {
        "statistical_analysis_score": 85,
        "data_visualization_score": 75,
        "python_score": 75,
        "sql_score": 80,
        "data_collection_score": 60,
        "data_quality_score": 80,
        "statistical_methods_score": 80,
        "communication_score": 70
    },
    "Data Analyst": {
        "statistical_analysis_score": 75,
        "data_visualization_score": 85,
        "python_score": 85,
        "sql_score": 85,
        "data_collection_score": 60,
        "data_quality_score": 80,
        "statistical_methods_score": 70,
        "communication_score": 75
    },
    "Junior Statistical Officer (JSO)": {
        "statistical_analysis_score": 70,
        "data_visualization_score": 65,
        "python_score": 60,
        "sql_score": 65,
        "data_collection_score": 85,
        "data_quality_score": 80,
        "statistical_methods_score": 75,
        "communication_score": 70
    },
    "Senior Statistical Officer (SSO)": {
        "statistical_analysis_score": 85,
        "data_visualization_score": 75,
        "python_score": 70,
        "sql_score": 75,
        "data_collection_score": 85,
        "data_quality_score": 90,
        "statistical_methods_score": 85,
        "communication_score": 80
    },
    "Field Operations Officer (NSSO)": {
        "statistical_analysis_score": 65,
        "data_visualization_score": 55,
        "python_score": 45,
        "sql_score": 50,
        "data_collection_score": 92,
        "data_quality_score": 85,
        "statistical_methods_score": 70,
        "communication_score": 85
    },
    "Assistant Director (Statistics)": {
        "statistical_analysis_score": 90,
        "data_visualization_score": 80,
        "python_score": 70,
        "sql_score": 75,
        "data_collection_score": 75,
        "data_quality_score": 85,
        "statistical_methods_score": 90,
        "communication_score": 85
    },
    "Survey Data Specialist": {
        "statistical_analysis_score": 80,
        "data_visualization_score": 70,
        "python_score": 65,
        "sql_score": 75,
        "data_collection_score": 88,
        "data_quality_score": 88,
        "statistical_methods_score": 80,
        "communication_score": 75
    }
}


def calculate_ground_truth_competency(avg_skill, assessment_score, exp_years):
    """
    Logically derives ground truth competency level:
    Weighted composite score:
    - 60% skills average
    - 30% comprehensive assessment score
    - 10% experience factor (capped at 100)
    """
    exp_factor = min(100, exp_years * 8.0)
    composite = 0.55 * avg_skill + 0.35 * assessment_score + 0.10 * exp_factor

    # Competency thresholds
    if composite < 50:
        return "Beginner"
    elif composite < 70:
        return "Developing"
    elif composite < 85:
        return "Proficient"
    else:
        return "Advanced"


def generate_official_training_dataset(num_records=1000, seed=42):
    """
    Generates a realistic synthetic dataset for official competency evaluation.
    """
    np.random.seed(seed)
    random.seed(seed)

    records = []

    for i in range(1, num_records + 1):
        emp_id = f"IND-OSS-{1000 + i}"
        role = random.choice(ROLES)
        department = random.choice(DEPARTMENTS)

        # Experience distributions (typical 1 - 25 years)
        exp_years = int(np.clip(np.random.gamma(shape=2.5, scale=2.5), 1, 28))

        # Base skill mean correlates with experience and role tier
        role_reqs = ROLE_SKILL_REQUIREMENTS[role]

        # Officials have individual proficiency tendencies
        proficiency_bias = np.random.normal(loc=0, scale=12)

        skill_scores = {}
        for skill in SKILL_NAMES:
            base_target = role_reqs[skill]
            # Actual score generated with realistic variation around target & experience
            exp_boost = min(15, exp_years * 0.8)
            score = base_target - 15 + proficiency_bias + exp_boost + np.random.normal(0, 7)
            # Clip between 15 and 99
            skill_scores[skill] = int(np.clip(round(score), 15, 99))

        # Training history
        prev_training_count = int(np.clip(np.random.poisson(lam=exp_years * 0.8 + 1), 0, 20))
        training_hours = prev_training_count * int(np.random.choice([15, 20, 30, 45, 60])) + int(np.random.randint(0, 20))

        # Assessment score correlates with skills + some test-day variation
        avg_skill = np.mean(list(skill_scores.values()))
        assessment_score = int(np.clip(round(avg_skill * 0.85 + np.random.normal(5, 6)), 20, 100))

        # Learning progress on current enrolled track (0-100%)
        learning_progress = int(np.clip(round(np.random.beta(a=2, b=2) * 100), 5, 98))

        # Target role (either current or aspirational next level)
        target_role = random.choice(ROLES)

        # Ground truth competency level
        competency_level = calculate_ground_truth_competency(avg_skill, assessment_score, exp_years)

        # Serialized requirements for target role
        req_skills_json = json.dumps(ROLE_SKILL_REQUIREMENTS[target_role])

        record = {
            "employee_id": emp_id,
            "role": role,
            "department": department,
            "experience_years": exp_years,
            "statistical_analysis_score": skill_scores["statistical_analysis_score"],
            "data_visualization_score": skill_scores["data_visualization_score"],
            "python_score": skill_scores["python_score"],
            "sql_score": skill_scores["sql_score"],
            "data_collection_score": skill_scores["data_collection_score"],
            "data_quality_score": skill_scores["data_quality_score"],
            "statistical_methods_score": skill_scores["statistical_methods_score"],
            "communication_score": skill_scores["communication_score"],
            "previous_training_count": prev_training_count,
            "training_hours_completed": training_hours,
            "assessment_score": assessment_score,
            "learning_progress": learning_progress,
            "target_role": target_role,
            "required_skill_levels": req_skills_json,
            "competency_level": competency_level,
            "data_type": "SYNTHETIC_PROTOTYPE"
        }
        records.append(record)

    df = pd.DataFrame(records)

    # Introduce a small number of realistic missing values (0.5%) to demonstrate data preprocessing resilience
    for col in ["previous_training_count", "learning_progress"]:
        mask = np.random.rand(len(df)) < 0.01
        df.loc[mask, col] = np.nan

    return df


def generate_training_catalogue():
    """
    Generates a rich, structured training course catalogue representing courses
    from iGOT Karmayogi (DoPT) and NSSTA (National Statistical Systems Training Academy, MoSPI).
    Clearly flagged as prototype catalogue entries.
    """
    catalogue = [
        # Python Courses
        {
            "course_id": "IGOT-PY-101",
            "course_name": "Python Foundations for Public Data Analysis",
            "skill": "python_score",
            "difficulty": "Beginner",
            "duration_hours": 20,
            "provider": "iGOT Karmayogi",
            "prerequisite": "Basic Computer Literacy",
            "description": "Introduction to Python data structures, pandas, and data cleaning tailored for civil servants working with survey tables.",
            "rating": 4.7
        },
        {
            "course_id": "NSSTA-PY-201",
            "course_name": "Applied Python for National Sample Survey Analysis",
            "skill": "python_score",
            "difficulty": "Developing",
            "duration_hours": 35,
            "provider": "NSSTA",
            "prerequisite": "Python Foundations (IGOT-PY-101)",
            "description": "Processing NSS microdata, handling large survey weights, and automating multi-stage tabulations using Python.",
            "rating": 4.9
        },
        {
            "course_id": "NSSTA-PY-301",
            "course_name": "Advanced Statistical Computing and Machine Learning with Python",
            "skill": "python_score",
            "difficulty": "Proficient",
            "duration_hours": 45,
            "provider": "NSSTA",
            "prerequisite": "Applied Python (NSSTA-PY-201)",
            "description": "Building predictive econometric models, nowcasting macroeconomic indicators, and anomaly detection in price index data.",
            "rating": 4.8
        },

        # SQL Courses
        {
            "course_id": "IGOT-SQL-101",
            "course_name": "Relational Databases and SQL Essentials for Governance",
            "skill": "sql_score",
            "difficulty": "Beginner",
            "duration_hours": 15,
            "provider": "iGOT Karmayogi",
            "prerequisite": "None",
            "description": "Fundamental SQL querying, filtering, table joins, and aggregations for government registry data management.",
            "rating": 4.6
        },
        {
            "course_id": "NSSTA-SQL-201",
            "course_name": "SQL for Statistical Data Warehousing and Aggregation",
            "skill": "sql_score",
            "difficulty": "Developing",
            "duration_hours": 25,
            "provider": "NSSTA",
            "prerequisite": "SQL Essentials (IGOT-SQL-101)",
            "description": "Complex subqueries, window functions, and indexing strategies for Census and Annual Survey of Industries (ASI) repositories.",
            "rating": 4.8
        },
        {
            "course_id": "IGOT-SQL-301",
            "course_name": "Enterprise Database Optimization & Pipeline Management",
            "skill": "sql_score",
            "difficulty": "Advanced",
            "duration_hours": 30,
            "provider": "iGOT Karmayogi",
            "prerequisite": "SQL for Statistical Data Warehousing",
            "description": "Query optimization, partitioning, automated ETL pipeline design, and data governance standards.",
            "rating": 4.7
        },

        # Statistical Analysis & Methods
        {
            "course_id": "NSSTA-STAT-101",
            "course_name": "Fundamentals of Official Statistics & Probability",
            "skill": "statistical_analysis_score",
            "difficulty": "Beginner",
            "duration_hours": 25,
            "provider": "NSSTA",
            "prerequisite": "Basic Mathematics",
            "description": "Core concepts of probability distributions, hypothesis testing, confidence intervals, and Indian statistical system framework.",
            "rating": 4.9
        },
        {
            "course_id": "NSSTA-STAT-201",
            "course_name": "Sampling Techniques & Multi-Stage Survey Design",
            "skill": "statistical_methods_score",
            "difficulty": "Developing",
            "duration_hours": 40,
            "provider": "NSSTA",
            "prerequisite": "Fundamentals of Official Statistics",
            "description": "Stratified random sampling, cluster sampling, PPS sampling, multiplier calculation, and variance estimation in survey rounds.",
            "rating": 5.0
        },
        {
            "course_id": "IGOT-STAT-301",
            "course_name": "Econometric Time Series Analysis & Forecasting",
            "skill": "statistical_analysis_score",
            "difficulty": "Proficient",
            "duration_hours": 35,
            "provider": "iGOT Karmayogi",
            "prerequisite": "Statistical Methods Intermediate",
            "description": "ARIMA, seasonal adjustment, unit root tests, and inflation / IIP trend forecasting methods.",
            "rating": 4.8
        },
        {
            "course_id": "NSSTA-STAT-401",
            "course_name": "National Accounts Statistics & SUT Framework",
            "skill": "statistical_methods_score",
            "difficulty": "Advanced",
            "duration_hours": 50,
            "provider": "NSSTA",
            "prerequisite": "Advanced Statistical Analysis",
            "description": "Compilation of GDP, GVA, Supply and Use Tables (SUT), System of National Accounts (SNA 2008), and institutional sector accounts.",
            "rating": 4.9
        },

        # Data Visualization
        {
            "course_id": "IGOT-VIZ-101",
            "course_name": "Data Storytelling and Dashboarding with Power BI / Tableau",
            "skill": "data_visualization_score",
            "difficulty": "Beginner",
            "duration_hours": 18,
            "provider": "iGOT Karmayogi",
            "prerequisite": "Basic Excel",
            "description": "Principles of executive dashboards, chart selection, and KPI cards for district and state progress monitoring.",
            "rating": 4.7
        },
        {
            "course_id": "NSSTA-VIZ-201",
            "course_name": "Geospatial Visualizations & Statistical Thematic Mapping",
            "skill": "data_visualization_score",
            "difficulty": "Developing",
            "duration_hours": 28,
            "provider": "NSSTA",
            "prerequisite": "Data Storytelling (IGOT-VIZ-101)",
            "description": "GIS mapping, choropleth maps, district-level indicator disaggregation, and interactive portal displays using QGIS and Python.",
            "rating": 4.8
        },
        {
            "course_id": "IGOT-VIZ-301",
            "course_name": "Interactive Statistical Infographics & Public Dissemination",
            "skill": "data_visualization_score",
            "difficulty": "Proficient",
            "duration_hours": 22,
            "provider": "iGOT Karmayogi",
            "prerequisite": "Dashboarding Foundations",
            "description": "Designing accessible web visualizations conforming to GIGW guidelines and open government data (data.gov.in) standards.",
            "rating": 4.6
        },

        # Data Collection & Field Operations
        {
            "course_id": "NSSTA-COL-101",
            "course_name": "Computer-Assisted Personal Interviewing (CAPI) Standards",
            "skill": "data_collection_score",
            "difficulty": "Beginner",
            "duration_hours": 20,
            "provider": "NSSTA",
            "prerequisite": "None",
            "description": "Digital questionnaire administration, tablet validation checks, GPS tagging, and ethical interview protocols in field surveys.",
            "rating": 4.9
        },
        {
            "course_id": "NSSTA-COL-201",
            "course_name": "Supervisory Controls & Rapid Field Audit Mechanisms",
            "skill": "data_collection_score",
            "difficulty": "Developing",
            "duration_hours": 25,
            "provider": "NSSTA",
            "prerequisite": "CAPI Standards (NSSTA-COL-101)",
            "description": "Back-checking methodologies, spot-checking protocols, and field officer feedback loops in large-scale socio-economic rounds.",
            "rating": 4.8
        },

        # Data Quality & Validation
        {
            "course_id": "NSSTA-QUAL-101",
            "course_name": "Statistical Data Editing, Imputation, & Scrutiny Protocols",
            "skill": "data_quality_score",
            "difficulty": "Beginner",
            "duration_hours": 22,
            "provider": "NSSTA",
            "prerequisite": "Fundamentals of Official Statistics",
            "description": "Logical scrutiny rules, outlier detection, cold-deck and hot-deck imputation techniques for official surveys.",
            "rating": 4.9
        },
        {
            "course_id": "IGOT-QUAL-201",
            "course_name": "National Data Governance & Quality Assurance Frameworks",
            "skill": "data_quality_score",
            "difficulty": "Proficient",
            "duration_hours": 30,
            "provider": "iGOT Karmayogi",
            "prerequisite": "Statistical Scrutiny Protocols",
            "description": "ISO and UN National Quality Assurance Frameworks (NQAF), metadata standards, and data confidentiality laws in India.",
            "rating": 4.7
        },

        # Communication & Dissemination
        {
            "course_id": "IGOT-COMM-101",
            "course_name": "Effective Statistical Report Writing & Policy Briefs",
            "skill": "communication_score",
            "difficulty": "Beginner",
            "duration_hours": 16,
            "provider": "iGOT Karmayogi",
            "prerequisite": "None",
            "description": "Translating quantitative statistical tables into concise policy memos, executive summaries, and press notes.",
            "rating": 4.8
        },
        {
            "course_id": "NSSTA-COMM-201",
            "course_name": "Stakeholder Dissemination & Statistical Consultation",
            "skill": "communication_score",
            "difficulty": "Developing",
            "duration_hours": 20,
            "provider": "NSSTA",
            "prerequisite": "Effective Statistical Report Writing",
            "description": "Conducting user-producer workshops, press conferences for key economic releases, and managing parliamentary queries.",
            "rating": 4.7
        }
    ]

    return pd.DataFrame(catalogue)


def generate_all_data(output_dir="data"):
    """
    Main helper to generate and save both CSV files.
    """
    os.makedirs(output_dir, exist_ok=True)

    df_officials = generate_official_training_dataset(num_records=1000)
    officials_path = os.path.join(output_dir, "official_training_dataset.csv")
    df_officials.to_csv(officials_path, index=False)
    print(f"Generated {len(df_officials)} synthetic official records at {officials_path}")

    df_catalogue = generate_training_catalogue()
    catalogue_path = os.path.join(output_dir, "training_catalogue.csv")
    df_catalogue.to_csv(catalogue_path, index=False)
    print(f"Generated {len(df_catalogue)} prototype training courses at {catalogue_path}")


if __name__ == "__main__":
    generate_all_data()
