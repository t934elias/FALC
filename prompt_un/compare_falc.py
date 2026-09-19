import json
import pandas as pd
import os
import numpy as np
import spacy
import textacy.extract
import textacy.preprocessing
import evaluate
from scipy.spatial.distance import cosine
from transformers import pipeline

# =========================
# CONFIG
# =========================
lang = "fr"
spacy_models = {"en": "en_core_web_md", "fr": "fr_core_news_md"}

# =========================
# LOAD DATA
# =========================
with open("results/falc_generes.json", "r", encoding="utf-8") as f:
    data = json.load(f)

rows = []
for entry in data:
    for run in entry["runs"]:
        rows.append({
            "sentence_id": entry["sentence_id"],
            "original_text": entry["original_text"],
            "ref_text": entry["falc"],          # competitor model output
            "my_text": run["summary"],          # your model output
            "temperature": run["parameters"]["temperature"],
            "num_ctx": run["parameters"]["num_ctx"],
        })

df = pd.DataFrame(rows)

# =========================
# NLP MODELS
# =========================
nlp = spacy.load(spacy_models[lang])

rouge = evaluate.load("rouge")
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")

classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None,
    device=-1
)

# =========================
# HELPERS
# =========================
def cosine_distance_vect(res1, res2):
    d1 = {x["label"]: x["score"] for x in res1}
    d2 = {x["label"]: x["score"] for x in res2}
    keys = sorted(set(d1) | set(d2))
    v1 = np.array([d1.get(k, 0) for k in keys])
    v2 = np.array([d2.get(k, 0) for k in keys])
    return cosine(v1, v2)

def compute_novelty(text, source):
    def uni(t):
        doc = nlp(textacy.preprocessing.remove.accents(t.lower()))
        return set(tok.text for tok in textacy.extract.ngrams(doc, 1))

    return len(uni(text) - uni(source)) / max(len(uni(text)), 1) * 100

# =========================
# DATA
# =========================
orig = df["original_text"].tolist()
my = df["my_text"].tolist()
ref = df["ref_text"].tolist()

# =========================
# EMOTION ANALYSIS
# =========================
emo_orig = classifier(orig, batch_size=16)
emo_my = classifier(my, batch_size=16)
emo_ref = classifier(ref, batch_size=16)

# =========================
# BERTSCORE
# =========================
bs_my = bertscore.compute(predictions=my, references=orig, lang=lang)
bs_ref = bertscore.compute(predictions=ref, references=orig, lang=lang)

# =========================
# MAIN LOOP METRICS
# =========================
results = []

for i in range(len(df)):

    src = orig[i]
    my_t = my[i]
    ref_t = ref[i]

    # -------- ROUGE --------
    rouge_my = rouge.compute(predictions=[my_t], references=[src])
    rouge_ref = rouge.compute(predictions=[ref_t], references=[src])

    # -------- BLEU --------
    bleu_my = bleu.compute(predictions=[my_t], references=[[src]])
    bleu_ref = bleu.compute(predictions=[ref_t], references=[[src]])

    # -------- BERTSCORE --------
    bs_my_f1 = bs_my["f1"][i] * 100
    bs_ref_f1 = bs_ref["f1"][i] * 100

    # -------- EMOTION SIM --------
    emo_my_sim = 1 - cosine_distance_vect(emo_orig[i], emo_my[i])
    emo_ref_sim = 1 - cosine_distance_vect(emo_orig[i], emo_ref[i])

    # -------- READABILITY (spaCy) --------
    doc_my = nlp(my_t)
    doc_ref = nlp(ref_t)
    doc_src = nlp(src)

    def fre(doc): return doc._.flesch_reading_ease
    def lix(doc): return doc._.lix
    def n_tokens(doc): return len(doc)

    # -------- NOVELTY --------
    nov_my = compute_novelty(my_t, src)
    nov_ref = compute_novelty(ref_t, src)

    # =========================
    # SCORES (IMPORTANT FIX)
    # =========================
    my_score = (
        0.35 * bs_my_f1 +
        0.25 * rouge_my["rougeL"] * 100 +
        0.15 * bleu_my["bleu"] * 100 +
        0.15 * emo_my_sim * 100 +
        0.10 * nov_my
    )

    ref_score = (
        0.35 * bs_ref_f1 +
        0.25 * rouge_ref["rougeL"] * 100 +
        0.15 * bleu_ref["bleu"] * 100 +
        0.15 * emo_ref_sim * 100 +
        0.10 * nov_ref
    )

    results.append({
        "sentence_id": df.iloc[i]["sentence_id"],

        # ---- BERT ----
        "bertscore_my": bs_my_f1,
        "bertscore_ref": bs_ref_f1,

        # ---- ROUGE ----
        "rougeL_my": rouge_my["rougeL"] * 100,
        "rougeL_ref": rouge_ref["rougeL"] * 100,

        # ---- BLEU ----
        "bleu_my": bleu_my["bleu"] * 100,
        "bleu_ref": bleu_ref["bleu"] * 100,

        # ---- EMOTION ----
        "emotion_my": emo_my_sim,
        "emotion_ref": emo_ref_sim,

        # ---- NOVELTY ----
        "novelty_my": nov_my,
        "novelty_ref": nov_ref,

        # ---- FINAL SCORE ----
        "score_my": my_score,
        "score_ref": ref_score,

        "winner": 1 if my_score > ref_score else 0
    })

# =========================
# SAVE
# =========================
df_res = pd.DataFrame(results)

os.makedirs("results", exist_ok=True)
df_res.to_csv("results/etr_comparison.csv", index=False)

# =========================
# SUMMARY
# =========================
win_rate = df_res["winner"].mean()

print(f"Done ✔")
print(f"Your model wins in {win_rate:.1%} of cases")