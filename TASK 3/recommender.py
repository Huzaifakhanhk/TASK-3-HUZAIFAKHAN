# ============================================================
#  DecodeLabs | AI Industrial Training Kit — Batch 2026
#  Project 3 : AI Recommendation Logic
#  Engine    : Tech Stack Recommender
#  Pipeline  : Ingest → TF-IDF → Cosine Similarity → Top-N
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────────────────────────────────
# DATASET — Job Roles with Required Skills
# (built-in, no external CSV needed)
# ─────────────────────────────────────────────
JOB_ROLES = [
    {
        "role": "Data Scientist",
        "skills": "python machine_learning deep_learning sql data_analysis statistics numpy pandas tensorflow scikit_learn visualization"
    },
    {
        "role": "Machine Learning Engineer",
        "skills": "python machine_learning deep_learning tensorflow pytorch neural_networks model_deployment mlops docker kubernetes scikit_learn"
    },
    {
        "role": "Data Engineer",
        "skills": "python sql spark hadoop kafka airflow etl pipeline data_warehousing cloud aws gcp azure"
    },
    {
        "role": "Backend Developer",
        "skills": "python java nodejs sql rest_api microservices docker kubernetes databases postgresql mongodb git"
    },
    {
        "role": "Frontend Developer",
        "skills": "javascript react vuejs html css nodejs typescript ui_ux responsive_design git figma"
    },
    {
        "role": "DevOps Engineer",
        "skills": "docker kubernetes aws azure gcp linux bash ci_cd jenkins terraform ansible monitoring git cloud"
    },
    {
        "role": "Cloud Architect",
        "skills": "aws azure gcp cloud infrastructure terraform kubernetes microservices security networking devops architecture"
    },
    {
        "role": "Cybersecurity Analyst",
        "skills": "networking linux security penetration_testing ethical_hacking firewalls encryption siem forensics risk_management compliance"
    },
    {
        "role": "Full Stack Developer",
        "skills": "javascript python react nodejs sql rest_api html css docker git databases mongodb postgresql"
    },
    {
        "role": "Business Intelligence Analyst",
        "skills": "sql data_analysis visualization power_bi tableau excel statistics reporting python pandas business_intelligence"
    },
    {
        "role": "NLP Engineer",
        "skills": "python nlp deep_learning transformers bert pytorch tensorflow text_processing huggingface linguistics machine_learning"
    },
    {
        "role": "Computer Vision Engineer",
        "skills": "python deep_learning pytorch tensorflow opencv image_processing convolutional_networks object_detection cuda gpu machine_learning"
    },
    {
        "role": "Site Reliability Engineer",
        "skills": "linux docker kubernetes monitoring logging aws cloud automation bash python ci_cd incident_management"
    },
    {
        "role": "Blockchain Developer",
        "skills": "solidity ethereum smart_contracts web3 javascript cryptography distributed_systems nodejs security defi"
    },
    {
        "role": "Mobile Developer",
        "skills": "flutter dart react_native swift kotlin android ios mobile_development api rest firebase git"
    },
    {
        "role": "Database Administrator",
        "skills": "sql postgresql mysql oracle mongodb database_design backup_recovery performance_tuning replication indexing linux"
    },
    {
        "role": "AI Research Engineer",
        "skills": "python deep_learning pytorch tensorflow machine_learning mathematics statistics linear_algebra research paper_writing experimentation"
    },
    {
        "role": "Automation Test Engineer",
        "skills": "selenium python java testing ci_cd jenkins git rest_api test_frameworks performance_testing agile"
    },
]

df_roles = pd.DataFrame(JOB_ROLES)

# ─────────────────────────────────────────────
# 1. DISPLAY WELCOME BANNER
# ─────────────────────────────────────────────
print("=" * 65)
print("  DecodeLabs — Project 3: AI Recommendation Logic")
print("  Tech Stack Recommender  |  Engine: TF-IDF + Cosine Similarity")
print("=" * 65)
print(f"\n  Dataset loaded: {len(df_roles)} Job Roles in the system")
print("\n  Available roles:")
for i, role in enumerate(df_roles['role'], 1):
    print(f"    {i:2}. {role}")

# ─────────────────────────────────────────────
# 2. STEP 1 — INGESTION: Take user input
#    (minimum 3 skills as required by the spec)
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 1 — INGESTION: Enter Your Skills")
print("─" * 65)
print("  Enter at least 3 skills you know (comma-separated).")
print("  Examples: python, machine_learning, sql, docker, aws,")
print("            react, nodejs, tensorflow, kubernetes, git")
print()

raw_input_str = input("  Your skills: ").strip()

# Parse and clean user skills
user_skills_raw = [s.strip().lower().replace(" ", "_")
                   for s in raw_input_str.split(",") if s.strip()]

if len(user_skills_raw) < 3:
    print("\n  [!] Fewer than 3 skills entered. Adding defaults to ensure")
    print("      sufficient data density for accurate matching.")
    user_skills_raw += ["python", "data_analysis", "sql"]
    user_skills_raw = list(dict.fromkeys(user_skills_raw))  # deduplicate

print(f"\n  Parsed skills : {user_skills_raw}")
user_profile_str = " ".join(user_skills_raw)

# ─────────────────────────────────────────────
# 3. STEP 2 — SCORING: TF-IDF + Cosine Similarity
# ─────────────────────────────────────────────
print("\n" + "─" * 65)
print("  STEP 2 — SCORING: Building TF-IDF Vector Space")
print("─" * 65)

# Combine all role skill documents + user profile into one corpus
all_documents = list(df_roles['skills']) + [user_profile_str]

# Fit TF-IDF on entire corpus (roles + user)
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_documents)

# Separate role vectors and user vector
role_vectors = tfidf_matrix[:-1]       # all rows except last
user_vector  = tfidf_matrix[-1]        # last row = user profile

vocab_size = len(vectorizer.get_feature_names_out())
print(f"  Vocabulary size  : {vocab_size} unique skill terms")
print(f"  User vector dims : {user_vector.shape[1]}")

# Compute cosine similarity between user and each role
similarity_scores = cosine_similarity(user_vector, role_vectors)[0]

print(f"\n  Raw similarity scores computed for {len(similarity_scores)} roles")

# ─────────────────────────────────────────────
# 4. STEP 3 — SORTING: Rank by similarity score
# ─────────────────────────────────────────────
df_roles['similarity_score'] = similarity_scores
df_sorted = df_roles.sort_values('similarity_score', ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────────
# 5. STEP 4 — FILTERING: Return Top-N results
# ─────────────────────────────────────────────
TOP_N = 5

print("\n" + "─" * 65)
print(f"  STEP 3 & 4 — SORTING & FILTERING: Top {TOP_N} Recommendations")
print("─" * 65)

print(f"\n  Your skill profile : {user_skills_raw}")
print(f"\n  {'Rank':<6} {'Job Role':<30} {'Match Score':<14} {'Alignment'}")
print("  " + "─" * 60)

for i, row in df_sorted.head(TOP_N).iterrows():
    score = row['similarity_score']
    pct   = score * 100
    bar   = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
    rank  = df_sorted.index.get_loc(i) + 1
    print(f"  #{rank:<5} {row['role']:<30} {pct:>6.1f}%       {bar}")

print("\n  Full Ranking:")
print(f"  {'Rank':<6} {'Job Role':<30} {'Score'}")
print("  " + "─" * 45)
for rank, (_, row) in enumerate(df_sorted.iterrows(), 1):
    print(f"  {rank:<6} {row['role']:<30} {row['similarity_score']:.4f}")

# ─────────────────────────────────────────────
# 6. VISUALIZATIONS — Save all plots
# ─────────────────────────────────────────────
colors_main = ['#1d3461', '#e85d04', '#4cc9f0', '#06d6a0', '#f4a261']
colors_all  = ['#1d3461' if i < TOP_N else '#cccccc' for i in range(len(df_sorted))]

# — Plot A: Top-N Recommendation Bar Chart
fig, ax = plt.subplots(figsize=(10, 6))
top_df = df_sorted.head(TOP_N)
bars = ax.barh(top_df['role'][::-1], top_df['similarity_score'][::-1],
               color=colors_main[:TOP_N][::-1], edgecolor='white', height=0.6)
for bar, val in zip(bars, top_df['similarity_score'][::-1]):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val*100:.1f}%', va='center', fontweight='bold', fontsize=11)
ax.set_title(f'Top {TOP_N} Career Recommendations\nSkills: {", ".join(user_skills_raw[:5])}',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Cosine Similarity Score (Match %)')
ax.set_xlim(0, max(top_df['similarity_score']) * 1.25)
ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig('plot_top_recommendations.png', dpi=150)
plt.close()

# — Plot B: Full ranking chart (all roles)
fig, ax = plt.subplots(figsize=(11, 8))
ax.barh(df_sorted['role'][::-1], df_sorted['similarity_score'][::-1],
        color=colors_all[::-1], edgecolor='white', height=0.6)
ax.axvline(df_sorted.iloc[TOP_N-1]['similarity_score'], color='#e85d04',
           linestyle='--', linewidth=1.5, label=f'Top-{TOP_N} cutoff')
ax.set_title('All Job Roles — Cosine Similarity Scores', fontsize=14, fontweight='bold')
ax.set_xlabel('Cosine Similarity Score')
ax.legend(); ax.grid(axis='x', alpha=0.3)
fig.tight_layout()
fig.savefig('plot_full_ranking.png', dpi=150)
plt.close()

# — Plot C: Skills overlap heatmap (Top 8 roles)
top8_roles  = df_sorted.head(8)['role'].tolist()
top8_skills = df_sorted.head(8)['skills'].tolist() + [user_profile_str]
top8_labels = top8_roles + ['[ YOU ]']

vec2 = TfidfVectorizer()
mat2 = vec2.fit_transform(top8_skills)
sim_matrix = cosine_similarity(mat2)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(sim_matrix, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=top8_labels, yticklabels=top8_labels,
            linewidths=0.5, ax=ax, annot_kws={'size': 9})
ax.set_title('Skill Similarity Heatmap — Top 8 Roles vs Your Profile',
             fontsize=13, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
fig.tight_layout()
fig.savefig('plot_similarity_heatmap.png', dpi=150)
plt.close()

# — Plot D: Score distribution
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df_sorted['similarity_score'], bins=15, color='#1d3461',
        edgecolor='white', alpha=0.85)
ax.axvline(df_sorted.iloc[0]['similarity_score'], color='#e85d04',
           linestyle='--', linewidth=2, label=f'Best match: {df_sorted.iloc[0]["role"]}')
ax.set_title('Distribution of Cosine Similarity Scores', fontsize=14, fontweight='bold')
ax.set_xlabel('Cosine Similarity Score')
ax.set_ylabel('Number of Roles')
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('plot_score_distribution.png', dpi=150)
plt.close()

print("\n[✓] All plots saved in the current folder.")
print("\n" + "=" * 65)
print("  Pipeline complete — Digital Matchmaker is ready.")
print(f"  Best match for your profile: {df_sorted.iloc[0]['role']}")
print(f"  Match score: {df_sorted.iloc[0]['similarity_score']*100:.1f}%")
print("=" * 65)