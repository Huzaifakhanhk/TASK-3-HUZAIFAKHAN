# DecodeLabs — AI Industrial Training Kit | Batch 2026
## Project 3: AI Recommendation Logic — Tech Stack Recommender

---

### 🎯 Goal
Build a content-based recommendation engine that maps a user's skills to the most relevant tech career paths using TF-IDF vectorization and Cosine Similarity.

---

### 📁 File Structure

```
project3_ai/
├── recommender.py                ← Main Python script (run this)
├── plot_top_recommendations.png  ← Top-N career matches bar chart
├── plot_full_ranking.png         ← All 18 roles ranked
├── plot_similarity_heatmap.png   ← Skill overlap heatmap
├── plot_score_distribution.png   ← Score distribution histogram
└── README.md                     ← This file
```

---

### ⚙️ How to Run

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
python recommender.py
```

When prompted, enter your skills comma-separated:
```
Your skills: python, machine_learning, sql, docker, aws
```

---

### 🧠 Pipeline (IPO Framework)

| Step | Phase | What Happens |
|------|-------|-------------|
| 1 | **Ingestion** | User enters ≥3 skills → parsed into profile string |
| 2 | **TF-IDF Vectorization** | All 18 roles + user profile → 120-dim numeric vectors |
| 3 | **Cosine Similarity Scoring** | User vector vs every role vector → similarity score |
| 4 | **Sorting** | Results ranked descending by score |
| 5 | **Filtering** | Top-N (default 5) returned to prevent choice overload |

---

### 🔑 Key Concepts Implemented

- **Content-Based Filtering** — matches user profile to item attributes, no other-user data needed
- **Vector Mapping** — qualitative skills → numerical TF-IDF vectors in shared vocabulary space
- **TF-IDF Weighting** — rare, specific skills get higher weight than common generic terms
- **Cosine Similarity** — measures angular alignment (direction), not magnitude → scale-invariant
- **Top-N Filtering** — prevents choice overload, outputs only highest-scoring matches
- **Cold Start Handling** — minimum 3 inputs enforced; defaults added if fewer provided

---

### 📊 Sample Output

```
Rank   Job Role                  Match Score    Alignment
#1     Machine Learning Engineer   32.8%       ██████░░░░░░░░░░░░░░
#2     Backend Developer           31.5%       ██████░░░░░░░░░░░░░░
#3     Data Scientist              31.3%       ██████░░░░░░░░░░░░░░
#4     Full Stack Developer        29.4%       █████░░░░░░░░░░░░░░░
#5     AI Research Engineer        18.3%       ███░░░░░░░░░░░░░░░░░
```

---

### 🗂️ Dataset — 18 Job Roles Built-In

Data Scientist, ML Engineer, Data Engineer, Backend Developer, Frontend Developer, DevOps Engineer, Cloud Architect, Cybersecurity Analyst, Full Stack Developer, BI Analyst, NLP Engineer, Computer Vision Engineer, SRE, Blockchain Developer, Mobile Developer, DBA, AI Research Engineer, Automation Test Engineer.

*No external CSV file needed — dataset is embedded in the script.*

---

*DecodeLabs | Powered by hands-on learning, not just theory.*
