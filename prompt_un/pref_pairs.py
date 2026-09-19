# import pandas as pd
# import numpy as np

# def get_median_idx(group):
#     median_val = group['score_dpo'].median()
#     return (group['score_dpo'] - median_val).abs().idxmin()

# df_falc = pd.read_excel('.\\resultat\\summaries_for_generation.xlsx')
# df_mesure = pd.read_excel('.\\resultat\\summaries_for_evaluation.xlsx')

# meilleur = df_mesure.groupby('sentence_id')['score_dpo'].idxmax()
# pire = df_mesure.groupby('sentence_id')['score_dpo'].idxmin()
# moyen = df_mesure.groupby('sentence_id').apply(get_median_idx)

# df_meilleur = df_falc.iloc[meilleur].copy()
# df_meilleur['score_dpo'] = df_mesure.loc[meilleur, 'score_dpo']
# df_pire = df_falc.iloc[pire].copy()
# df_pire['score_dpo'] = df_mesure.loc[pire, 'score_dpo']
# df_moyen = df_falc.iloc[moyen].copy()
# df_moyen['score_dpo'] = df_mesure.loc[moyen, 'score_dpo'].values

# # print(df_meilleur)
# # print(df_pire)

# df_meilleur = df_meilleur.rename(columns={
#     'original_text': 'prompt', 
#     'falc': 'chosen'
# })

# df_pire = df_pire.rename(columns={
#     'falc': 'rejected'
# })

# df_meilleur = df_meilleur[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
# df_pire = df_pire[['sentence_id', 'rejected', 'score_dpo_rejected']]

# df_dpo = pd.merge(df_meilleur, df_pire, on='sentence_id')

# final_dpo_columns = ['prompt', 'chosen', 'rejected', 'score_dpo_chosen', 'score_dpo_rejected']
# df_dpo = df_dpo[final_dpo_columns]

# # print(df_dpo.head())
# print(df_dpo)

# df_dpo.to_json('.\\resultat\\dpo_dataset.jsonl', orient='records', lines=True, force_ascii=False)



# # import pandas as pd

# # df_dpo_loaded = pd.read_json('dpo_dataset.jsonl', orient='records', lines=True)

# # print(df_dpo_loaded.head())



























# import pandas as pd
# import numpy as np

# df_falc = pd.read_csv('../results/summaries_for_generation_2.csv')
# df_mesure = pd.read_csv('../results/summaries_for_evaluation_2.csv')

# meilleur = df_mesure.groupby('sentence_id')['score_dpo'].idxmax()
# pire = df_mesure.groupby('sentence_id')['score_dpo'].idxmin()

# def get_median_idx(group):
#     median_val = group['score_dpo'].median()
#     return (group['score_dpo'] - median_val).abs().idxmin()

# moyen = df_mesure.groupby('sentence_id').apply(get_median_idx)

# df_meilleur = df_falc.iloc[meilleur].copy()
# df_meilleur['score_dpo'] = df_mesure.loc[meilleur, 'score_dpo'].values

# df_moyen = df_falc.iloc[moyen].copy()
# df_moyen['score_dpo'] = df_mesure.loc[moyen, 'score_dpo'].values

# df_pire = df_falc.iloc[pire].copy()
# df_pire['score_dpo'] = df_mesure.loc[pire, 'score_dpo'].values

# df_p1_chosen = df_meilleur.rename(columns={'original_text': 'prompt', 'falc': 'chosen', 'score_dpo': 'score_dpo_chosen'})[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
# df_p1_rejected = df_moyen.rename(columns={'falc': 'rejected', 'score_dpo': 'score_dpo_rejected'})[['sentence_id', 'rejected', 'score_dpo_rejected']]

# df_pair_best_avg = pd.merge(df_p1_chosen, df_p1_rejected, on='sentence_id')

# df_p2_chosen = df_moyen.rename(columns={'original_text': 'prompt', 'falc': 'chosen', 'score_dpo': 'score_dpo_chosen'})[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
# df_p2_rejected = df_pire.rename(columns={'falc': 'rejected', 'score_dpo': 'score_dpo_rejected'})[['sentence_id', 'rejected', 'score_dpo_rejected']]

# df_pair_avg_worst = pd.merge(df_p2_chosen, df_p2_rejected, on='sentence_id')

# df_meilleur = df_meilleur.rename(columns={'original_text': 'prompt', 'falc': 'chosen', 'score_dpo': 'score_dpo_chosen'})[['sentence_id', 'prompt', 'chosen', 'score_dpo_chosen']]
# df_pire = df_pire.rename(columns={'falc': 'rejected', 'score_dpo': 'score_dpo_rejected'})[['sentence_id', 'rejected', 'score_dpo_rejected']]
# df_pair_best_worst = pd.merge(df_meilleur, df_pire, on='sentence_id')

# df_dpo = pd.concat([df_pair_best_worst, df_pair_best_avg, df_pair_avg_worst], ignore_index=True)

# df_dpo = df_dpo[df_dpo['chosen'] != df_dpo['rejected']]

# final_dpo_columns = ['prompt', 'chosen', 'rejected', 'score_dpo_chosen', 'score_dpo_rejected']
# df_dpo = df_dpo[final_dpo_columns]

# print(f"Total preference pairs generated: {len(df_dpo)}")
# # print(df_dpo.head())

# df_dpo.to_json('../results/dpo_dataset_2.jsonl', orient='records', lines=True, force_ascii=False)




import pandas as pd
from itertools import combinations

# -----------------------------
# Parameters
# -----------------------------
SCORE_THRESHOLD = 10      # Keep only pairs where chosen_score - rejected_score >= threshold

# -----------------------------
# Load data
# -----------------------------
df_falc = pd.read_csv("../results/summaries_for_generation_2.csv")
df_mesure = pd.read_csv("../results/summaries_for_evaluation_2.csv")

# Merge scores with summaries
df = df_falc.copy()
df["score_dpo"] = df_mesure["score_dpo"]

# -----------------------------
# Build all pairwise preferences
# -----------------------------
pairs = []

for sentence_id, group in df.groupby("sentence_id"):

    # Highest score first
    group = group.sort_values("score_dpo", ascending=False).reset_index(drop=True)

    rows = group.to_dict("records")

    # Every ordered pair (better, worse)
    for better, worse in combinations(rows, 2):

        score_gap = better["score_dpo"] - worse["score_dpo"]

        # Skip weak preferences
        if score_gap < SCORE_THRESHOLD:
            continue

        # Skip identical summaries
        if better["falc"].strip() == worse["falc"].strip():
            continue

        pairs.append({
            "prompt": better["original_text"],
            "chosen": better["falc"],
            "rejected": worse["falc"],
            "score_dpo_chosen": better["score_dpo"],
            "score_dpo_rejected": worse["score_dpo"],
            "score_gap": score_gap
        })

# -----------------------------
# Create dataframe
# -----------------------------
df_dpo = pd.DataFrame(pairs)

# Remove duplicate preference pairs
df_dpo = df_dpo.drop_duplicates(
    subset=["prompt", "chosen", "rejected"]
)

# Shuffle
df_dpo = df_dpo.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Total DPO pairs: {len(df_dpo)}")

print("\nScore gap statistics:")
print(df_dpo["score_gap"].describe())
print(f"df length {len(df_dpo)}")

# Save
df_dpo.to_json(
    "../results/dpo_dataset_2.jsonl",
    orient="records",
    lines=True,
    force_ascii=False,
)