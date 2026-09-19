import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# 1. Load the data
# Make sure to save your runs as separate files!
file_dpo = "results/falc_comp_dpo.json"
file_standard = "results/falc_comp_standard.json"

with open(file_dpo, "r", encoding="utf-8") as f:
    data_dpo = json.load(f)

with open(file_standard, "r", encoding="utf-8") as f:
    data_standard = json.load(f)


# 2. Extract and Parse Metrics
rows = []

for dpo_entry, std_entry in zip(data_dpo, data_standard):
    sent_id = dpo_entry["sentence_id"]

    orig_text = dpo_entry["original_text"]
    # Extract DPO metrics
    d_run = dpo_entry["runs"][0]
    d_metrics = d_run["metrics"]
    # Calculate an average emotional distance from origin across emotions
    d_em_dist = np.mean(list(d_run["dis_ori_pred"].values())) if d_run["dis_ori_pred"] else 0
    d_cos = np.mean(d_run["cosine_ori_pred"]) if d_run["cosine_ori_pred"] else 0
    d_pred = d_run["summary"]
    
    # Extract Standard metrics
    s_run = std_entry["runs"][0]
    s_metrics = s_run["metrics"]
    s_em_dist = np.mean(list(s_run["dis_ori_pred"].values())) if s_run["dis_ori_pred"] else 0
    s_cos = np.mean(s_run["cosine_ori_pred"]) if s_run["cosine_ori_pred"] else 0
    s_pred = s_run["summary"]

    d_row = {
        "sentence_id": sent_id,
        "model": "em_phrase_dpo",
        "original_text":orig_text,
        "pred":d_pred,
        "rougeL": d_metrics.get("rougeL", 0),
        "bert_f1": d_metrics.get("bertscore_f1", 0),
        "sari": d_metrics.get("sari", 0),
        "srb": d_metrics.get("srb", 0),
        "lix": d_metrics.get("lix", 0),  # lower is easier to read
        "compression": d_metrics.get("compression_ratio", 0),
        "emotion_distance": d_cos # lower is closer emotionally
    }
    # Dynamically inject individual emotion scores (prefixed with 'emo_')
    if d_run["dis_ori_pred"]:
        for emo_name, emo_val in d_run["dis_ori_pred"].items():
            d_row[f"emo_{emo_name}"] = emo_val
            
    rows.append(d_row)


    s_row = {
        "sentence_id": sent_id,
        "model": "em_phrase",
        "original_text":orig_text,
        "pred":s_pred,
        "rougeL": s_metrics.get("rougeL", 0),
        "bert_f1": s_metrics.get("bertscore_f1", 0),
        "sari": s_metrics.get("sari", 0),
        "srb": s_metrics.get("srb", 0),
        "lix": s_metrics.get("lix", 0),
        "compression": s_metrics.get("compression_ratio", 0),
        "emotion_distance": s_cos
    }
    # Dynamically inject individual emotion scores
    if s_run["dis_ori_pred"]:
        for emo_name, emo_val in s_run["dis_ori_pred"].items():
            s_row[f"emo_{emo_name}"] = emo_val
    rows.append(s_row)

df = pd.DataFrame(rows)

# 3. Print Statistical Summary
summary = df.groupby("model").mean(numeric_only=True).drop(columns=["sentence_id"])
print("\n=== Average Performance Metrics (Comparison) ===")
print(summary.to_string())

df_metrics = df.drop(columns=["sentence_id", "original_text", "pred"])
summary_metrics = df_metrics.groupby("model").agg(["mean", "min", "max"])

print("\n=== Performance Metrics Summary ===")
print(summary_metrics.T.to_string())


# Determine the winner dynamically
better_readability = summary["srb"].idxmax()
closer_emotion = summary["emotion_distance"].idxmin()
print(f"\nBest overall FALC Simplification (SRB Score): {better_readability}")
print(f"Best emotional fidelity (Lowest Emotion Distance): {closer_emotion}")

# 4. Data Visualization
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot A: FALC Overarching Quality (SRB Score)
sns.boxplot(
    data=df, x="model", y="srb", ax=axes[0, 0], 
    palette="Set2", hue="model", legend=False
)
axes[0, 0].set_title("FALC Overall Quality (SRB Score - Higher is Better)")
axes[0, 0].set_ylabel("SRB (Harmonic Mean of SARI, ROUGE-L, BERTScore)")

# Plot B: Emotional Preservation
sns.boxplot(
    data=df, x="model", y="emotion_distance", ax=axes[0, 1], 
    palette="Set2", hue="model", legend=False
)
axes[0, 1].set_title("Emotional Distance from Original (Lower is Better)")
axes[0, 1].set_ylabel("Mean Absolute Difference in Emotion Vectors")

# Plot C: SARI score (Simplification Quality)
sns.barplot(
    data=df, x="model", y="sari", ax=axes[1, 0], 
    palette="Set2", hue="model", legend=False, errorbar=None
)
axes[1, 0].set_title("Average SARI Score")
axes[1, 0].set_ylabel("SARI Score")

# Plot D: Readability Ease vs Emotion Distance Scatter
sns.scatterplot(
    data=df, x="lix", y="emotion_distance", hue="model", 
    style="model", s=100, palette="Set2", ax=axes[1, 1]
)
axes[1, 1].set_title("Readability (LIX) vs Emotional Distance")
axes[1, 1].set_xlabel("LIX Readability Index (Lower is Easier)")
axes[1, 1].set_ylabel("Emotion Distance")

plt.tight_layout()
plt.savefig("model_comparison_plots.png", dpi=300)
print("\n📊 Plots generated successfully and saved to 'results/model_comparison_plots.png'!")
plt.show()



# 5. Sentence-Level Head-to-Head Comparison
# Pick the first two unique sentence IDs to compare
target_sentences = df["sentence_id"].unique()[:2]
df_subset = df[df["sentence_id"].isin(target_sentences)]

# Melt the dataframe to make it easy for Seaborn to map individual metrics
df_melted = pd.melt(
    df_subset, 
    id_vars=["sentence_id", "model"], 
    value_vars=["rougeL", "bert_f1", "sari", "srb", "lix", "compression", "emotion_distance"],
    var_name="Metric", 
    value_name="Score"
)

# Create a grid of charts—one for each metric
g = sns.catplot(
    data=df_melted,
    x="sentence_id", 
    y="Score", 
    hue="model", 
    col="Metric", 
    col_wrap=4,          # Wrap after 4 charts to keep it tidy
    kind="bar", 
    palette="Set2",
    sharey=False,        # Crucial! Gives each metric its own independent scale
    height=3.5, 
    aspect=1.2
)

# Clean up titles and layout
g.set_titles("{col_name}")
g.set_axis_labels("Sentence ID", "Value")
g.fig.subplots_adjust(top=0.88)
g.fig.suptitle("Sentence-Level Comparison: em_phrase vs em_phrase_dpo (First Two Sentences)", fontsize=16, weight="bold")

plt.savefig("sentence_comparison_plots.png", dpi=300)
print("\n🎯 Sentence-level comparison plots saved to 'sentence_comparison_plots.png'!")
plt.show()

# 6. Emotion-Specific Head-to-Head Comparison
target_sentences = df["sentence_id"].unique()[:2]
df_subset = df[df["sentence_id"].isin(target_sentences)]

# Find all column names that start with 'emo_'
emotion_cols = [col for col in df.columns if col.startswith("emo_")]

# Melt only the individual emotions
df_emotions_melted = pd.melt(
    df_subset, 
    id_vars=["sentence_id", "model"], 
    value_vars=emotion_cols,
    var_name="Emotion", 
    value_name="Distance"
)

# Clean up labels (removes 'emo_' prefix from titles for presentation)
df_emotions_melted["Emotion"] = df_emotions_melted["Emotion"].str.replace("emo_", "")

# Generate the plot
g_emo = sns.catplot(
    data=df_emotions_melted,
    x="sentence_id", 
    y="Distance", 
    hue="model", 
    col="Emotion", 
    col_wrap=4, 
    kind="bar", 
    palette="Set2",
    sharey=True,        # Keep y-axis shared so you can visually see which emotion shifted the most
    height=3.5, 
    aspect=1.2
)

g_emo.set_axis_labels("Sentence ID", "Distance from Original")
g_emo.fig.subplots_adjust(top=0.85)
g_emo.fig.suptitle("Sentence-Level Emotion Profile: em_phrase vs em_phrase_dpo", fontsize=16, weight="bold")

plt.savefig("emotion_profile_comparison.png", dpi=300)
print("🎭 Emotion-specific breakdown plots saved to 'emotion_profile_comparison.png'!")
plt.show()


# 7. Find and Compare the Best Sentences
print("\n==================================================")
print("🔍 HEAD-TO-HEAD BEST PERFORMING PHRASES")
print("==================================================")

# List of models we want to evaluate
models = ["em_phrase", "em_phrase_dpo"]

for target_model in models:
    # 1. Isolate target model and find row with max SRB score
    model_df = df[df["model"] == target_model]
    best_row = model_df.loc[model_df["srb"].idxmax()]
    best_sentence_id = best_row["sentence_id"]
    
    # 2. Grab both models' outputs for this specific sentence
    comparison_df = df[df["sentence_id"] == best_sentence_id]
    
    print(f"\n🏆 Best Phrase for [{target_model.upper()}] (Sentence ID: {best_sentence_id})")
    print(f"   Max SRB Score: {best_row['srb']:.4f}")
    print("-" * 65)
    
    # Print the text comparison
    for idx, row in comparison_df.iterrows():
        prefix = "👉" if row["model"] == target_model else "🔗"
        print(f"{prefix} [{row['model']}]:")
        print(f"original text: {row['original_text']}")
        print(f"falc pred: {row['pred']}")
        print(f"   [Metrics] -> SRB: {row['srb']:.4f} | SARI: {row['sari']:.1f} | LIX: {row['lix']:.1f} | Emo-Dist: {row['emotion_distance']:.4f}")
        print()
    print("=" * 65)






# 8. Phrase-by-Phrase Win Count Analysis
print("\n==================================================")
print("📊 PHRASE-BY-PHRASE WIN-LOSS BREAKDOWN")
print("==================================================")

# Split the data to compare head-to-head per sentence
dpo_side = df[df["model"] == "em_phrase_dpo"].set_index("sentence_id")
std_side = df[df["model"] == "em_phrase"].set_index("sentence_id")

# Align indexes to ensure we only compare matching sentence IDs
common_ids = dpo_side.index.intersection(std_side.index)
dpo_side = dpo_side.loc[common_ids]
std_side = std_side.loc[common_ids]

total_phrases = len(common_ids)

# --- Metric 1: SRB (Higher is Better) ---
dpo_wins_srb = (dpo_side["srb"] > std_side["srb"]).sum()
std_wins_srb = (std_side["srb"] > dpo_side["srb"]).sum()
ties_srb = (dpo_side["srb"] == std_side["srb"]).sum()

# --- Metric 2: LIX Readability (Lower is Better/Easier) ---
dpo_wins_lix = (dpo_side["lix"] < std_side["lix"]).sum()
std_wins_lix = (std_side["lix"] < dpo_side["lix"]).sum()
ties_lix = (dpo_side["lix"] == std_side["lix"]).sum()

# --- Metric 3: Emotion Distance (Lower is Better/Closer) ---
dpo_wins_emo = (dpo_side["emotion_distance"] < std_side["emotion_distance"]).sum()
std_wins_emo = (std_side["emotion_distance"] < dpo_side["emotion_distance"]).sum()
ties_emo = (dpo_side["emotion_distance"] == std_side["emotion_distance"]).sum()

# Print results
print(f"Analyzing {total_phrases} paired phrases:\n")

print(f"⭐ OVERALL QUALITY (SRB Score - Higher is Better)")
print(f"   • em_phrase_dpo was better for: {dpo_wins_srb} phrases ({dpo_wins_srb/total_phrases*100:.1f}%)")
print(f"   • em_phrase was better for:     {std_wins_srb} phrases ({std_wins_srb/total_phrases*100:.1f}%)")
print(f"   • Tied phrases:                 {ties_srb} phrases ({ties_srb/total_phrases*100:.1f}%)")
print()

print(f"📖 READABILITY (LIX Index - Lower/Easier is Better)")
print(f"   • em_phrase_dpo was better for: {dpo_wins_lix} phrases ({dpo_wins_lix/total_phrases*100:.1f}%)")
print(f"   • em_phrase was better for:     {std_wins_lix} phrases ({std_wins_lix/total_phrases*100:.1f}%)")
print(f"   • Tied phrases:                 {ties_lix} phrases ({ties_lix/total_phrases*100:.1f}%)")
print()

print(f"🎭 EMOTIONAL FIDELITY (Distance - Lower/Closer is Better)")
print(f"   • em_phrase_dpo was better for: {dpo_wins_emo} phrases ({dpo_wins_emo/total_phrases*100:.1f}%)")
print(f"   • em_phrase was better for:     {std_wins_emo} phrases ({std_wins_emo/total_phrases*100:.1f}%)")
print(f"   • Tied phrases:                 {ties_emo} phrases ({ties_emo/total_phrases*100:.1f}%)")
print("==================================================")






# --- Overall Phrase-by-Phrase Champion (Combining all 3 metrics) ---
dpo_overall_wins = 0
std_overall_wins = 0
overall_ties = 0

for idx in common_ids:
    d_row = dpo_side.loc[idx]
    s_row = std_side.loc[idx]
    
    # Track votes for this specific phrase
    dpo_votes = 0
    std_votes = 0
    
    # Vote 1: Quality (SRB - Higher is better)
    if d_row["srb"] > s_row["srb"]:
        dpo_votes += 1
    elif s_row["srb"] > d_row["srb"]:
        std_votes += 1
        
    # Vote 2: Readability (LIX - Lower is better)
    if d_row["lix"] < s_row["lix"]:
        dpo_votes += 2
    elif s_row["lix"] < d_row["lix"]:
        std_votes += 2
        
    # Vote 3: Emotional Fidelity (Lower distance is better)
    if d_row["emotion_distance"] < s_row["emotion_distance"]:
        dpo_votes += 3
    elif s_row["emotion_distance"] < d_row["emotion_distance"]:
        std_votes += 3
        
    # Determine the champion for this phrase
    if dpo_votes > std_votes:
        dpo_overall_wins += 1
    elif std_votes > dpo_votes:
        std_overall_wins += 1
    else:
        overall_ties += 1

print(f"🏆 OVERALL PHRASE CHAMPION (Model winning the majority of the 3 dimensions)")
print(f"   • em_phrase_dpo was better overall for: {dpo_overall_wins} phrases ({dpo_overall_wins/total_phrases*100:.1f}%)")
print(f"   • em_phrase was better overall for:     {std_overall_wins} phrases ({std_overall_wins/total_phrases*100:.1f}%)")
print(f"   • Tied overall:                         {overall_ties} phrases ({overall_ties/total_phrases*100:.1f}%)")
print("==================================================")