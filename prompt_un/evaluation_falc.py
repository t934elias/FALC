import json
import pandas as pd
import os
import ctypes
import nltk
import numpy as np
import spacy
import textacy.extract
import textacy.preprocessing
import textdescriptives as td
import evaluate
from scipy.stats import hmean
from scipy.spatial.distance import cosine
from transformers import pipeline

# # --- Environment and Logging Setup ---
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# # Silence the verbose transformer load reports causing the repetitions
# os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# try:
#     ctypes.CDLL(r"E:\python_envs\etr_env\Lib\site-packages\torch\lib\libiomp5md.dll")
# except Exception as e:
#     print(f"Manual load failed: {e}")

nltk.download("punkt", quiet=True)

# Metric Key Constants
METRIC_KEY_ROUGE = "rouge"
METRIC_KEY_ROUGEL = "rougeL"
METRIC_KEY_BLEU = "bleu"
METRIC_KEY_BERTSCORE_F1 = "bertscore_f1"
METRIC_KEY_KMRE = "fre"
METRIC_KEY_LIX = "lix"
METRIC_KEY_COMPRESSION = "compression_ratio"
METRIC_KEY_NOVELTY = "novelty"
METRIC_KEY_SRB = "srb"
METRIC_KEY_SARI = "sari"

# --- Load Data ---
with open("../results/falc_generes_2.json", "r", encoding="utf-8") as f:
    data = json.load(f)

flattened_rows = []
for entry in data:
    orig_id = entry["sentence_id"]
    orig_text = entry["original_text"]
    falc_text = entry["falc"]
    
    for run in entry["runs"]:
        row = {
            "sentence_id": orig_id,
            "original_text": orig_text,
            "falc_text": falc_text,
            "temperature": run["parameters"]["temperature"],
            "num_ctx": run["parameters"]["num_ctx"],
            "falc": run["summary"]
        }
        flattened_rows.append(row)

df = pd.DataFrame(flattened_rows)

# --- Initialize Global Models & Metrics (Loaded ONCE) ---
lang = "fr"
spacy_models = {"en": "en_core_web_md", "fr": "fr_core_news_md"}
nlp = spacy.load(spacy_models[lang])
nlp.add_pipe("textdescriptives/readability")

bleu_metric = evaluate.load("sacrebleu")
rouge_metric = evaluate.load("rouge")
sari_metric = evaluate.load("sari")
bertscore_metric = evaluate.load("bertscore")

classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None,
    device=-1 # Set to 0 if you install torch with CUDA and upgrade versions
)

# --- Helper Vector Functions ---

def cosine_distance_vect(res1, res2):
    d1 = {x["label"]: x["score"] for x in res1}
    d2 = {x["label"]: x["score"] for x in res2}
    keys = sorted(set(d1) | set(d2))
    v1 = np.array([d1.get(k, 0) for k in keys])
    v2 = np.array([d2.get(k, 0) for k in keys])
    return cosine(v1, v2)

def compute_novelty_score(source, prediction):
    def unigram(text):
        parsed = nlp(textacy.preprocessing.remove.accents(text.lower()))
        return set([span.text for span in textacy.extract.ngrams(parsed, 1, filter_nums=True)])
    
    src_unigrams = unigram(source)
    pred_unigrams = unigram(prediction)
    if not pred_unigrams:
        return 0.0
    return (len(pred_unigrams - src_unigrams) / len(pred_unigrams)) * 100

# --- Vectorized Processing Pipeline ---

predictions = df['falc'].astype(str).tolist()
references = df['falc_text'].astype(str).tolist()
sources = df['original_text'].astype(str).tolist()

# 1. Pipeline Classifier Batch Execution (Massive speed up)
results_pred_all = classifier(predictions, batch_size=16)
results_ori_all = classifier(sources, batch_size=16)
results_ref_all = classifier(references, batch_size=16)
# results_pred_all = classifier(predictions, batch_size=16, truncation=True, max_length=512)
# results_ori_all = classifier(sources, batch_size=16, truncation=True, max_length=512)
# results_ref_all = classifier(references, batch_size=16, truncation=True, max_length=512)

# 2. Batch Metric Computations
bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang=lang)
# sari_results = sari_metric.compute(predictions=predictions, references=[[r] for r in references], sources=sources)

# Readability Batching via SpaCy Pipe
docs_preds = list(nlp.pipe(predictions))
docs_sources = list(nlp.pipe(sources))

metrics_preds = td.extract_dict(docs_preds, include_text=False)
metrics_sources = td.extract_dict(docs_sources, include_text=False)

# --- Reconstruct Results Matrix ---
resultats = []

for idx in range(len(df)):
    current_pred = predictions[idx]
    current_ref = references[idx]
    current_src = sources[idx]

    row_sari_output = sari_metric.compute(
        predictions=[current_pred],
        references=[[current_ref]],
        sources=[current_src]
    )
    sari_val = row_sari_output["sari"]

    # Readability values
    n_tokens_pred = metrics_preds[idx]["n_tokens"]
    n_tokens_src = metrics_sources[idx]["n_tokens"]
    comp_ratio = 100 - (n_tokens_pred / n_tokens_src * 100) if n_tokens_src > 0 else 0
    
    fre_score = metrics_preds[idx]["flesch_reading_ease"]
    lix_score = metrics_preds[idx]["lix"]
    
    # Emotion vector distances
    sim_ori_pred = 1 - cosine_distance_vect(results_ori_all[idx], results_pred_all[idx])
    sim_ori_ref = 1 - cosine_distance_vect(results_ori_all[idx], results_ref_all[idx])
    sim_pred_ref = 1 - cosine_distance_vect(results_pred_all[idx], results_ref_all[idx])
    
    # Novelty
    nov_score = compute_novelty_score(sources[idx], predictions[idx])
    
    # Extract structural components
    bs_f1 = bertscore_results["f1"][idx] * 100
    bs_prec = bertscore_results["precision"][idx] * 100
    bs_rec = bertscore_results["recall"][idx] * 100
    
    bert_avg = (bs_prec + bs_rec + bs_f1) / 3
    
    # Custom Reward DPO Equation Setup derived from your specifications
    # Adjusted to conform with normal 0-100 scales across variables
    score_dpo = (0.2 * sari_val) + (0.3 * bert_avg) + (0.1 * fre_score) + (0.05 * nov_score) + (0.05 * comp_ratio) + (0.25 * sim_ori_pred * 100)

    metric_row = {
        "sentence_id": df.iloc[idx]["sentence_id"],
        "df_index": idx,
        "sari": round(sari_val, 6),
        "bertscore_precision": round(bs_prec, 6),
        "bertscore_recall": round(bs_rec, 6),
        METRIC_KEY_BERTSCORE_F1: round(bs_f1, 6),
        METRIC_KEY_KMRE: round(fre_score, 6),
        METRIC_KEY_LIX: round(lix_score, 6),
        METRIC_KEY_COMPRESSION: round(comp_ratio, 6),
        METRIC_KEY_NOVELTY: round(nov_score, 6),
        "dis_ori_pred": round(sim_ori_pred, 6),
        "dis_ori_ref": round(sim_ori_ref, 6),
        "dis_pred_ref": round(sim_pred_ref, 6),
        "score_dpo": round(score_dpo, 6)
    }
    resultats.append(metric_row)

df_res = pd.DataFrame(resultats)

# --- Save Records Out ---
os.makedirs("results", exist_ok=True)
# df.to_excel("results/summaries_for_generation.xlsx", index=False)
# df_res.to_excel("results/summaries_for_evaluation.xlsx", index=False)

df.to_csv("../results/summaries_for_generation_2.csv", index=False)
df_res.to_csv("../results/summaries_for_evaluation_2.csv", index=False)
