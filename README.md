# AI-Powered Statistical Competency Assessment & Personalized Learning Platform
### Smart India Hackathon 2026 – Problem Statement SIH26101
**Team:** Dream Weavers (AD14)  
**Target Domain:** India's Official Statistical System (MoSPI, NSSO, CSO, NSSTA, State DES)

---

## 📌 1. Problem Overview
India's Official Statistical System—encompassing the Ministry of Statistics and Programme Implementation (MoSPI), National Sample Survey Office (NSSO), Central Statistics Office (CSO), and State Directorates of Economics and Statistics (DES)—relies heavily on the specialized skills of statistical officers. However, skill requirements rapidly evolve with the advent of big data, CAPI (Computer-Assisted Personal Interviewing), automated ETL pipelines, and predictive analytics.

### Solution:
A fully functional AI/ML platform that:
1. **Predicts Current Competency Level** (Beginner, Developing, Proficient, Advanced) from multi-dimensional official profiles.
2. **Detects & Quantifies Skill Gaps** against role-specific benchmark standards.
3. **Ranks Gap Priorities** with transparent, explainable rationales.
4. **Recommends Personalized Training Courses** from accredited bodies like **iGOT Karmayogi** and **NSSTA**.
5. **Constructs Multi-Phase Personalized Learning Paths**.
6. **Generates Domain-Specific AI Quizzes / MCQs** on identified skill deficits.
7. **Evaluates Quiz Performance & Dynamically Updates Competency** with Before/After progression tracking.

---

## 🔬 2. Synthetic Dataset (`official_training_dataset.csv`)

> **DISCLAIMER:**  
> This dataset contains **1,000 synthetic prototype records** generated for academic and demonstration purposes. It does **NOT** contain real personal data or confidential government records.

### Field Dictionary:
| Column | Type | Description |
| :--- | :--- | :--- |
| `employee_id` | String | Unique official identifier (e.g. `IND-OSS-1001`) |
| `role` | Categorical | Official designation (Statistical Analyst, JSO, SSO, etc.) |
| `department` | Categorical | Division (CSO, NSSO, DQAD, ESD, NSSTA, DES) |
| `experience_years` | Numeric | Years of active service in the statistical system |
| `statistical_analysis_score` | Numeric (0-100) | Inference, hypothesis testing, index numbers |
| `data_visualization_score` | Numeric (0-100) | Dashboards, thematic maps, chart selection |
| `python_score` | Numeric (0-100) | Pandas, numpy, automated survey processing |
| `sql_score` | Numeric (0-100) | Database queries, joins, window functions |
| `data_collection_score` | Numeric (0-100) | CAPI administration, sampling frame verification |
| `data_quality_score` | Numeric (0-100) | Outlier detection, scrutiny, hot/cold deck imputation |
| `statistical_methods_score` | Numeric (0-100) | Multi-stage stratified sampling, variance estimation |
| `communication_score` | Numeric (0-100) | Policy brief writing, stakeholder dissemination |
| `previous_training_count` | Numeric | Number of completed in-service training programmes |
| `training_hours_completed` | Numeric | Cumulative official training hours |
| `assessment_score` | Numeric (0-100) | Comprehensive standardized assessment score |
| `learning_progress` | Numeric (0-100) | Active course completion percentage |
| `target_role` | Categorical | Aspirational or benchmark role |
| `competency_level` | Target Class | Ground truth: `Beginner`, `Developing`, `Proficient`, `Advanced` |

---

## ⚙️ 3. Data Preprocessing Pipeline (`src/data_preprocessing.py`)

The preprocessing engine handles raw profiles and prepares feature matrices for both training and real-time inference:

1. **Duplicate Detection & Removal:** Deduplicates records based on `employee_id`.
2. **Missing-Value Imputation:**
   - Numerical columns: Median imputation (`SimpleImputer(strategy='median')`).
   - Categorical columns: Mode imputation (`SimpleImputer(strategy='most_frequent')`).
3. **Categorical Feature Encoding:** One-Hot Encoding (`handle_unknown='ignore'`) for `role`, `department`, and `target_role`.
4. **Numerical Feature Scaling:** Z-score normalization (`StandardScaler`) on all 13 numeric features.
5. **Stratified Train/Test Split:** 80% Train (800 records), 20% Test (200 records), maintaining class distribution balance.
6. **Artifact Persistence:** Serialized to `models/preprocessing.pkl`.

---

## 🤖 4. Competency Assessment Model (`src/competency_model.py`)

We trained and compared two distinct machine learning paradigms on identical preprocessed train/test partitions:

1. **Random Forest Classifier** (`n_estimators=150`, `max_depth=12`, `min_samples_split=4`, `class_weight='balanced'`)
2. **Multinomial Logistic Regression** (`max_iter=1000`, `C=1.0`, `class_weight='balanced'`)

### Actual Evaluation Results (Factual Metrics):

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | F1-Score (Weighted) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | **91.00%** | **0.9170** | **0.8567** | **0.8774** | **0.9091** | **SELECTED BEST** |
| Logistic Regression | 91.50% | 0.8403 | 0.8647 | 0.8514 | 0.9159 | Candidate |

> **Selection Rationale:**  
> While both models attained ~91% accuracy, **Random Forest** achieved a significantly superior **Macro F1-Score (0.8774 vs 0.8514)** and **Macro Precision (0.9170 vs 0.8403)**, demonstrating balanced capability across all four competency tiers without bias towards majority classes.

Saved model artifact: `models/competency_model.pkl`  
Confusion matrices and comparison plots: `models/plots/`

---

## 📊 5. Skill Gap Detection & Explainability (`src/skill_gap.py`)

For any given official and target role, the engine compares:
$$\text{Skill Gap} = \text{Required Benchmark Score} - \text{Current Official Score}$$

### Gap Classification Tiers:
- **No Gap ($\le 0$ points):** Official meets or exceeds benchmark. Priority: `None`.
- **Low Gap ($1 - 15$ points):** Minor deficit. Priority: `Low`. Recommended: Self-paced micro-modules.
- **Medium Gap ($16 - 30$ points):** Moderate deficit. Priority: `Medium`. Recommended: Formal structured course & hands-on lab.
- **High Gap ($> 30$ points):** Critical deficit. Priority: `High`. Recommended: Mandatory intensive training with mentorship.

### Model Explainability:
Every deficit is accompanied by human-interpretable reasoning:
> *"Python for Statistical Computing was identified as a High Priority Gap because current score is 38/100 while the required competency level for 'Assistant Director (Statistics)' is 70/100 (critical deficit of 32 points)."*

---

## 🎯 6. Personalized Training Recommendation Engine (`src/recommendation.py`)

Recommendations are derived from the prototype training catalogue (`data/training_catalogue.csv`) covering:
- **iGOT Karmayogi** (Digital civil service capacity building)
- **NSSTA** (National Statistical Systems Training Academy, Greater Noida)

### Logic:
1. **Gap Alignment:** Courses targeting the official's highest priority deficits are ranked first.
2. **Difficulty Calibration:** Course difficulty (`Beginner`, `Developing`, `Proficient`, `Advanced`) is matched against the official's predicted competency tier.
3. **Multi-Phase Learning Path:** Generates a structured roadmap:
   - **Phase 1:** Critical Competency Rectification (High Priority gaps)
   - **Phase 2:** Core Analytical Capability Building (Medium Priority gaps)
   - **Phase 3:** Domain Specialization & Quality Standards (Low/Expansion gaps)

---

## 🧠 7. AI Quiz Generation & Evaluation (`src/quiz_generator.py`)

- **Clean Abstraction Layer:** Supports dynamic generation via LLM API (Gemini / OpenAI) when configured.
- **Deterministic Offline Fallback:** When running offline or without an API key, an authentic domain question bank on Indian Official Statistics (MoSPI, NSSO, CAPI, Index numbers, Python, SQL) guarantees 100% demo uptime.
- **Interactive Evaluation:** Grades submissions, generates question-by-question explanations, calculates percentage score, awards competency boosts (+2 to +8 points), and re-evaluates the official's competency tier.

---

## 👥 8. Verified Demo Profiles (`src/demo_profiles.py`)

1. **Employee A: Dr. Rajesh Sharma (`IND-OSS-0101`)**
   - *Current:* Statistical Analyst (CSO) $\rightarrow$ *Target:* Senior Statistical Officer (SSO)
   - *Profile:* PhD in Economics; high statistical methods (84) but moderate Python (42) and SQL (52).
2. **Employee B: Pooja Verma (`IND-OSS-0202`)**
   - *Current:* Data Analyst (DQAD) $\rightarrow$ *Target:* Statistical Analyst
   - *Profile:* Computer science background; high Python (88) and SQL (84) but lacks sampling methods (54) and CAPI collection (48).
3. **Employee C: Amitabh Sen (`IND-OSS-0303`)**
   - *Current:* Junior Statistical Officer (NSSO) $\rightarrow$ *Target:* Assistant Director (Statistics)
   - *Profile:* 8 years field experience; exceptional CAPI (92) and quality scrutiny (86), seeking promotion requiring advanced Python (38 vs 70 req) and modeling.

---

## 🚀 9. How to Run the Project

### Prerequisites
- Python 3.10+ (tested on Python 3.14)
- Pip

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Synthetic Dataset & Training Catalogue
```bash
python src/data_generation.py
```

### Step 3: Train and Evaluate Machine Learning Models
```bash
python src/competency_model.py
```
*(This trains Random Forest & Logistic Regression, evaluates them, selects the best model, and saves plots to `models/plots/`)*.

### Step 4: Start the FastAPI Server
```bash
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Step 5: Open the Interactive Web Dashboard
Open your browser and navigate to:
```
http://127.0.0.1:8000/
```
From here you can:
- Switch between Demo Officials A, B, and C with one click.
- View the Competency Radar Chart and Skill Gap Deficit Bar Chart.
- Explore recommended iGOT / NSSTA courses and learning paths.
- Take interactive quizzes and see the live competency progression chart!

---

## 🌐 10. API Endpoints & Usage

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server status and active model metadata |
| `GET` | `/demo-profiles` | List summary of all demo official profiles |
| `GET` | `/demo-profiles/{key}` | Retrieve single demo profile (`employee_a`, `employee_b`, `employee_c`) |
| `POST` | `/assess` | Official profile $\rightarrow$ Competency level, gaps, radar chart |
| `POST` | `/recommend` | Profile + gaps $\rightarrow$ iGOT / NSSTA course recommendations & roadmap |
| `POST` | `/generate-quiz` | Learning material $\rightarrow$ 5 to 10 AI-generated MCQs |
| `POST` | `/evaluate-quiz` | Quiz answers $\rightarrow$ Score, skill boost, updated competency level |
| `GET` | `/evaluation-report` | Factual model evaluation metrics & comparison JSON |

### Sample cURL Commands:

#### 1. Assess Official:
```bash
curl -X POST "http://127.0.0.1:8000/assess" \
     -H "Content-Type: application/json" \
     -d @- <<EOF
{
  "employee_id": "IND-OSS-0101",
  "role": "Statistical Analyst",
  "department": "CSO",
  "target_role": "Senior Statistical Officer (SSO)",
  "experience_years": 4,
  "statistical_analysis_score": 86,
  "data_visualization_score": 68,
  "python_score": 42,
  "sql_score": 52,
  "data_collection_score": 65,
  "data_quality_score": 82,
  "statistical_methods_score": 84,
  "communication_score": 74
}
EOF
```

#### 2. Generate Quiz:
```bash
curl -X POST "http://127.0.0.1:8000/generate-quiz" \
     -H "Content-Type: application/json" \
     -d '{
       "learning_material": "NSSO sample surveys and Python pandas",
       "num_questions": 5,
       "target_skill": "python_score",
       "difficulty": "Developing"
     }'
```

---

## ⚠️ 11. Limitations & Future Integrations

1. **Synthetic Data Disclaimer:** The current dataset was realistically modeled to simulate Indian statistical official profiles. In production, this can be ingested from official HRMS / SPARROW portals with proper data governance.
2. **Prototype Training Catalogue:** Course entries are modeled after real iGOT Karmayogi and NSSTA curricula. In production, this connects to the live **iGOT Karmayogi Open API** (via DoPT OAuth2) and NSSTA training portal.
3. **CAPI Tablet Integration:** In future phases, offline quiz results and on-the-job CAPI scrutiny metrics can feed directly from field tablets into the learning platform.
