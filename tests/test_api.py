"""
Automated Integration Tests for SIH 2026 Problem Statement SIH26101
Team: Dream Weavers (AD14)
"""
import requests
import json
import sys

BASE_URL = "http://127.0.0.1:8000"

def test_system():
    print("Running system validation tests against", BASE_URL)

    # 1. Health
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200, f"Health check failed: {r.status_code}"
    health = r.json()
    assert health["model_loaded"] is True
    print("[PASS] GET /health (Model: " + health["active_model"] + ")")

    # 2. Demo Profiles
    r = requests.get(f"{BASE_URL}/demo-profiles")
    assert r.status_code == 200
    profiles = r.json()
    assert len(profiles) >= 3
    print(f"[PASS] GET /demo-profiles ({len(profiles)} profiles loaded)")

    # 3. Assessment
    emp = requests.get(f"{BASE_URL}/demo-profiles/employee_a").json()
    r = requests.post(f"{BASE_URL}/assess", json=emp)
    assert r.status_code == 200
    assess = r.json()
    assert "competency_assessment" in assess
    assert "skill_gap_analysis" in assess
    print(f"[PASS] POST /assess (Predicted: {assess['competency_assessment']['predicted_competency']})")

    # 4. Recommendation
    rec_req = {
        "official_profile": emp,
        "competency_level": assess["competency_assessment"]["predicted_competency"]
    }
    r = requests.post(f"{BASE_URL}/recommend", json=rec_req)
    assert r.status_code == 200
    recs = r.json()
    assert len(recs["recommended_courses"]) > 0
    print(f"[PASS] POST /recommend ({len(recs['recommended_courses'])} courses recommended)")

    # 5. Quiz Generation
    r = requests.post(f"{BASE_URL}/generate-quiz", json={
        "learning_material": "NSSO multi-stage sampling and Python",
        "num_questions": 5,
        "target_skill": "python_score"
    })
    assert r.status_code == 200
    quiz = r.json()
    assert len(quiz["questions"]) == 5
    print(f"[PASS] POST /generate-quiz ({len(quiz['questions'])} MCQs generated)")

    # 6. Quiz Evaluation
    answers = {q["id"]: q["correct_answer"] for q in quiz["questions"]}
    r = requests.post(f"{BASE_URL}/evaluate-quiz", json={
        "official_profile": emp,
        "quiz_questions": quiz["questions"],
        "submitted_answers": answers
    })
    assert r.status_code == 200
    ev_res = r.json()
    assert ev_res["quiz_evaluation"]["score_percentage"] == 100.0
    print(f"[PASS] POST /evaluate-quiz (Score: 100%, Transition: {ev_res['previous_competency']} -> {ev_res['updated_competency']})")

    # 7. Evaluation Report
    r = requests.get(f"{BASE_URL}/evaluation-report")
    assert r.status_code == 200
    rep = r.json()
    assert "Random Forest" in rep["models_evaluated"]
    print(f"[PASS] GET /evaluation-report (Best Model: {rep['best_model']})")

    print("\n>>> ALL TESTS PASSED! PRODUCTION READY. <<<")

if __name__ == "__main__":
    test_system()
