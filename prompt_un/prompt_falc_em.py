import json
from ollama import chat
import math
from transformers import pipeline
import os
import ctypes
from functools import reduce
import nltk
import numpy as np
import spacy
import textacy.extract
import textacy.preprocessing
import textdescriptives as td
import evaluate
from scipy.stats import hmean
import csv
from scipy.spatial.distance import cosine

# def get_feel_ultra_light():
#     emo_map = {'joy': 'j', 'sadness': 's', 'anger': 'a', 'fear': 'f', 'disgust': 'd', 'surprise': 'u'}
    
#     compact_list = []
    
#     with open('FEEL.csv', newline='', encoding='utf-8') as csvfile:
#         reader = csv.DictReader(csvfile, delimiter=';')
#         for row in reader:
#             word = row.pop('word')
#             codes = "".join(emo_map[e] for e in emo_map if row.get(e) == '1')
            
#             if codes:
#                 compact_list.append(f"{word}:{codes}")

#     return "|".join(compact_list)

# data = []

# with open('..\data\etr-fr\sources\\test.json', 'r') as file:
#     for line in file:
#         if line.strip(): 
#             data.append(json.loads(line))

# src = data[2]['original']
# ref = data[2]['falc']

src = "L\u00e9a, pourquoi es-tu triste ? Ce matin, Cl\u00e9ment se r\u00e9veille tout content. Aujourd'hui, c'est l'anniversaire de sa petite s\u0153ur L\u00e9a. Encore en pyjama, il saute de son lit et se pr\u00e9cipite vers la chambre de L\u00e9a. Il est impatient de lui souhaiter son anniversaire. Arriv\u00e9 devant sa porte, il frappe doucement avant d'entrer en criant joyeusement: \u00ab Bon anniversaire, petite s\u0153ur ! \u00bb L\u00e9a, encore un peu endormie, ouvre un \u0153il. La petite fille a eu du mal \u00e0 trouver le sommeil cette nuit. C'est pourquoi, ce matin elle est vraiment tr\u00e8s fatigu\u00e9e. Elle aurait voulu se reposer encore un peu. \u00c0 la place, son fr\u00e8re n'arr\u00eate pas de sautiller devant elle en r\u00e9p\u00e9tant gaiement : \u00ab Tu es grande maintenant! \u00bb."
ref = "L\u00e9a est triste Ce matin, Cl\u00e9ment est tout content. C'est l'anniversaire de sa s\u0153ur L\u00e9a. Cl\u00e9ment entre dans la chambre de L\u00e9a. \u00ab Bon anniversaire, petite s\u0153ur ! Tu as 8 ans aujourd'hui. Tu es grande maintenant! \u00bb"


examples = []

with open('./data/train.json', 'r') as file:
    for line in file:
        if line.strip(): 
            examples.append(json.loads(line))


def build_system_prompt():
    return f"""
Tu es un expert de simplification FALC (Facile à Lire et à Comprendre). Le texte final doit etre facile à comprendre pour les personnes ayant des difficultés de compréhension (handicap mental, dyslexie, personnes âgées, apprenants du français, etc.) tout en gardant l'idee principale du texte original.

Règles de simplification FALC :
- Garde uniquement les informations essentielles.
- Respecte le sens exact du texte original.
- Présente les informations dans un ordre logique.
- Mets l'idée principale au début.
- Utilise des phrases courtes et simples.
- Choisis des mots faciles.
- Explique les mots difficiles.
- Utilise un langage clair et direct.
- Évite les idées abstraites et inutiles au sujet.


Consignes de sortie : 
- Réponds uniquement avec le texte réécrit en FALC. 
- Ne donne aucune explication. 
- N'ajoute pas de titre. 
- N'ajoute pas de section. 
- N'ajoute pas de commentaire. 
- Ne fais pas de liste. 
- Garde le même contexte et le même message que le texte original. 
"""


def build_em():
    return f"""
     Tu es un expert de la simplification émotionnelle des textes. Ton objectif est d'ajouter a un texte falc les memes emotions que celle dans le texte original. Le texte final doit rester facile à comprendre, comme le falc donne, pour les personnes ayant des difficultés de compréhension (handicap mental, dyslexie, personnes âgées, apprenants du français, etc.) tout en gardant l'émotion du texte original très présente et naturelle.
    Règles émotionnelles très importantes :
    - Identifie l'émotion principale du texte : joie, tristesse, peur, colère, surprise ou neutre.
    - Identifie aussi l'intensité de cette émotion : faible, moyenne ou forte.
    - Rajoute cette émotion au texte falc simplifié donne.
    - Utiliser des mots émotionnels simples pour montrer cette émotion.
    - Fais ressentir clairement l'émotion dans chaque phrase importante.

    Méthode obligatoire :
    1. Lire le texte original entier attentivement.
    2. Associer les phrases du texte original aux phrases du texte FALC.
    3. Identifier l'émotion de chaque phrase et son intensité dans le texte.
    4. Choisir des mots simples qui transmettent cette émotion.
    5. Réécrire le texte en FALC en ajoutant les émotions visibles et naturelles.

    Consignes de sortie : 
    - Réponds uniquement avec le texte réécrit. 
    - Ne donne aucune explication. 
    - N'ajoute pas de titre. 
    - N'ajoute pas de section. 
    - N'ajoute pas de commentaire. 
    - Ne fais pas de liste. 
    - Tu dois répondre UNIQUEMENT en français.

    """


# def build_em():
#     return f"""
#      Tu es un expert de la simplification émotionnelle des textes. Ton objectif est d'ajouter a un texte falc les memes emotions que celle dans le texte original. Le texte final doit rester facile à comprendre, comme le falc donne, pour les personnes ayant des difficultés de compréhension (handicap mental, dyslexie, personnes âgées, apprenants du français, etc.) tout en gardant l'émotion du texte original très présente et naturelle.
#     Règles émotionnelles très importantes :
#     - Identifie l'émotion principale du texte : joie, tristesse, peur, colère, surprise ou neutre.
#     - Identifie aussi l'intensité de cette émotion : faible, moyenne ou forte.
#     - Garde cette émotion tout au long du texte falc simplifié.
#     - Utiliser des mots émotionnels simples pour montrer cette émotion.
#     - Fais ressentir clairement l'émotion dans chaque phrase importante.

#     Méthode obligatoire :
#     1. Lire le texte entier attentivement.
#     2. Identifier le sujet principal.
#     3. Identifier l'émotion de chaque phrase et son intensité.
#     4. Choisir des mots simples qui transmettent cette émotion.
#     5. Réécrire le texte en FALC en gardant les émotions visibles et naturelles.

#     Consignes de sortie : 
#     - Réponds uniquement avec le texte réécrit. 
#     - Ne donne aucune explication. 
#     - N'ajoute pas de titre. 
#     - N'ajoute pas de section. 
#     - N'ajoute pas de commentaire. 
#     - Ne fais pas de liste. 
#     - Tu dois répondre UNIQUEMENT en français.

#     """

few_shot_messages = []

for ex in examples[:5]:
    few_shot_messages.extend([
        {
            "role": "user",
            "content": f"Réécris ce texte en FALC : {ex['original']}"
        },
        {
            "role": "assistant",
            "content": ex['falc']
        }
    ])

response = chat(
    # model='mistral:latest',
    model = 'llama3:latest',
    messages=[
        {"role": "system", "content": build_system_prompt()},
        *few_shot_messages,
        {"role": "user", "content": f"Réécris ce texte en FALC : {src}"}
    ],
    options={
        "num_ctx": 12288
    }
)

falc = response['message']['content']

print(f"\nFALC predit : \n{falc}")

# -------- SECOND CALL (emotion injection) --------
response2 = chat(
    model='mistral:latest',
    messages=[
        {"role": "system", "content": build_em()},
        {
            "role": "user",
            "content": f"Texte original : {src}\n\nTexte FALC : {falc}"
        }
    ],
    options={
        "num_ctx": 12288
    }
)

pred = response2['message']['content']

print(f"\nTexte original : \n{src}")
print(f"\nFALC avec emotions predit : \n{pred}")
print(f"\nFALC ref : \n{ref}")



os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    ctypes.CDLL(r"E:\python_envs\etr_env\Lib\site-packages\torch\lib\libiomp5md.dll")
except Exception as e:
    print(f"Manual load failed: {e}")

nltk.download("punkt")

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


def load_nlp(lang="fr"):
    spacy_models = {
        "en": "en_core_web_md",
        "fr": "fr_core_news_md"
    }
    nlp = spacy.load(spacy_models[lang])
    nlp.add_pipe("textdescriptives/readability")

    return nlp

def compute_readability_scores(texts, nlp):

    docs = nlp.pipe(texts)

    metrics_list = td.extract_dict(
        docs,
        include_text=False
    )

    merged_dict = reduce(
        lambda x, y: {**x, **y},
        metrics_list
    )

    return {
        k: [merged_dict[k] for merged_dict in metrics_list]
        for k in merged_dict
    }

def compute_bleu(predictions, references):

    bleu_metric = evaluate.load("sacrebleu")

    preds = [p.strip() for p in predictions]

    refs = [
        [r[0].strip()] if isinstance(r, list)
        else [r.strip()]
        for r in references
    ]

    result = bleu_metric.compute(
        predictions=preds,
        references=refs
    )

    return {
        METRIC_KEY_BLEU: result["score"]
    }

def compute_rouge(predictions, references):

    rouge_metric = evaluate.load("rouge")

    def preprocess(texts):
        return [
            "\n".join(nltk.sent_tokenize(text.strip()))
            for text in texts
        ]

    preds = preprocess(predictions)

    refs = preprocess([
        r[0] if isinstance(r, list) else r
        for r in references
    ])

    result = rouge_metric.compute(
        predictions=preds,
        references=refs,
        use_stemmer=True
    )

    return {
        k: round(v * 100, 4)
        for k, v in result.items()
    }

def compute_sari(predictions, references, sources):

    sari_metric = evaluate.load("sari")

    preds = [p.strip() for p in predictions]

    refs = [
        [r[0].strip()] if isinstance(r, list)
        else [r.strip()]
        for r in references
    ]

    srcs = [s.strip() for s in sources]

    result = sari_metric.compute(
        predictions=preds,
        references=refs,
        sources=srcs
    )

    return {
        METRIC_KEY_SARI: result["sari"]
    }

def compute_bertscore(predictions, references, lang="fr"):

    bertscore_metric = evaluate.load("bertscore")

    preds = [p.strip() for p in predictions]

    refs = [
        r[0].strip() if isinstance(r, list)
        else r.strip()
        for r in references
    ]

    result = bertscore_metric.compute(
        predictions=preds,
        references=refs,
        lang=lang
    )

    return {
        "bertscore_precision":
            np.mean(result["precision"]) * 100,

        "bertscore_recall":
            np.mean(result["recall"]) * 100,

        METRIC_KEY_BERTSCORE_F1:
            np.mean(result["f1"]) * 100,
    }

def compute_readability(predictions, sources, nlp):

    preds = [p.strip() for p in predictions]
    srcs = [s.strip() for s in sources]

    scores_preds = compute_readability_scores(
        preds,
        nlp
    )

    scores_sources = compute_readability_scores(
        srcs,
        nlp
    )

    def compression_ratio(ntok_src, ntok_tgt):
        return 100 - (
            ntok_tgt / ntok_src * 100
        )

    # print(scores_preds)
    return {
        

        METRIC_KEY_KMRE:
            np.mean(
                scores_preds["flesch_reading_ease"]
            ),

        METRIC_KEY_LIX:
            np.mean(
                scores_preds["lix"]
            ),

        METRIC_KEY_COMPRESSION:
            np.mean(
                compression_ratio(
                    np.array(scores_sources["n_tokens"]),
                    np.array(scores_preds["n_tokens"])
                )
            ),
    }

def compute_novelty(predictions, sources, nlp):

    def unigram(text):

        text = nlp(
            textacy.preprocessing.remove.accents(
                text.lower()
            )
        )

        return set([
            span.text
            for span in textacy.extract.ngrams(
                text,
                1,
                filter_nums=True
            )
        ])

    def novelty(source, prediction):

        src_unigrams = unigram(source)
        pred_unigrams = unigram(prediction)

        try:
            return (
                len(pred_unigrams - src_unigrams)
                / len(pred_unigrams)
                * 100
            )

        except:
            return 0.0

    scores = [
        novelty(s, p)
        for s, p in zip(sources, predictions)
    ]

    return {
        METRIC_KEY_NOVELTY: np.mean(scores)
    }

def compute_srb(sari, rougeL, bertscore_f1):

    return {
        METRIC_KEY_SRB:
            hmean([
                sari,
                rougeL,
                bertscore_f1
            ])
    }

def evaluate_text(
    predictions,
    references,
    sources=None,
    lang="fr"
):

    results = {}
    nlp = load_nlp(lang)

    results.update(
        compute_bleu(
            predictions,
            references
        )
    )

    results.update(
        compute_rouge(
            predictions,
            references
        )
    )

    if sources is not None:

        results.update(
            compute_sari(
                predictions,
                references,
                sources
            )
        )

    results.update(
        compute_bertscore(
            predictions,
            references,
            lang
        )
    )

    if sources is not None:

        results.update(
            compute_readability(
                predictions,
                sources,
                nlp
            )
        )

    if sources is not None:

        results.update(
            compute_novelty(
                predictions,
                sources,
                nlp
            )
        )

    if sources is not None:

        results.update(
            compute_srb(
                results[METRIC_KEY_SARI],
                results[METRIC_KEY_ROUGEL],
                results[METRIC_KEY_BERTSCORE_F1]
            )
        )

    results["n_samples"] = len(predictions)

    final_results = {}

    for k, v in results.items():

        if isinstance(v, (float, np.floating)):
            final_results[k] = round(v, 4)
        else:
            final_results[k] = v

    return final_results

metrics = evaluate_text(
    predictions=[pred],
    references=[[ref]],
    sources=[src],
    lang="fr"
)

print(f"\nBLEU Score: {metrics['bleu']:.4f}")

print(f"ROUGE-L: {metrics['rougeL']:.4f}")

print(f"SARI Score: {metrics['sari']:.4f}")

print("\nBERTScore:")
print(f"Precision: {metrics['bertscore_precision']:.4f}")
print(f"Recall:    {metrics['bertscore_recall']:.4f}")
print(f"F1:        {metrics['bertscore_f1']:.4f}")

print("\nReadability:")
print(f"flesch reading ease: {metrics['fre']:.4f}")
print(f"LIX:                 {metrics['lix']:.4f}")
print(f"Compression Ratio:   {metrics['compression_ratio']:.4f}")

print("\nNovelty:")
print(f"Novelty Score: {metrics['novelty']:.4f}")

print("\nCombined Metric:")
print(f"SRB Score: {metrics['srb']:.4f}")

print(f"\nNumber of Samples: {metrics['n_samples']}")


metrics_pred_vs_ref = evaluate_text(
    predictions=[pred],
    references=[[ref]],
    sources=[src],
    lang="fr"
)

# Pair 2: How close is your model (Pred) to the raw, unsimplified text (Src)?
metrics_pred_vs_src = evaluate_text(
    predictions=[pred],
    references=[[src]],
    sources=None, # Set to None to avoid SARI/Readability loop errors here
    lang="fr"
)

# Pair 3: How close is the human gold standard (Ref) to the raw text (Src)?
metrics_ref_vs_src = evaluate_text(
    predictions=[ref],
    references=[[src]],
    sources=None,
    lang="fr"
)

# Print Comparison Matrix
print("\n" + "="*50)
print("             NLP SCORE COMPARISON MATRIX")
print("="*50)
print(f"{'Metric':<15} | {'Pred vs Ref':<12} | {'Pred vs Src':<12} | {'Ref vs Src':<12}")
print("-"*50)

for metric_key, display_name in [('bleu', 'BLEU'), ('rougeL', 'ROUGE-L'), ('bertscore_f1', 'BERTScore F1')]:
    val_pred_ref = metrics_pred_vs_ref.get(metric_key, 0.0)
    val_pred_src = metrics_pred_vs_src.get(metric_key, 0.0)
    val_val_ref_src = metrics_ref_vs_src.get(metric_key, 0.0)
    print(f"{display_name:<15} | {val_pred_ref:<12.4f} | {val_pred_src:<12.4f} | {val_val_ref_src:<12.4f}")

print("="*50)



classifier = pipeline(
    "text-classification",
    model="astrosbd/french_emotion_camembert",
    top_k=None
)

def euclidean_distance(res1, res2):
    vec1 = {item['label']: item['score'] for item in res1[0]}
    vec2 = {item['label']: item['score'] for item in res2[0]}

    # union de toutes les émotions
    labels = set(vec1.keys()) | set(vec2.keys())

    sum_squared = 0

    for label in labels:
        x = vec1.get(label, 0)
        y = vec2.get(label, 0)

        sum_squared += (x - y) ** 2

    return math.sqrt(sum_squared)

def compute_distance(res1, res2):
    distance = {}

    res2_map = {item['label']: item['score'] for item in res2[0]}
    
    for r in res1[0]:
        label = r['label']
        if label in res2_map:
            distance[label] = math.sqrt((r['score'] - res2_map[label]) ** 2)
            
    return distance

def cosinie_distance(res1, res2):
    vect2 = np.array([item['score'] for item in res2[0]])
    vect1 = np.array([item['score'] for item in res1[0]])
    distance = cosine(vect1, vect2)

    return distance

results_pred = classifier(pred)
results_ori = classifier(src)
results_ref = classifier(ref)

print("\nTexte original :")
for r in results_ori[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nTexte FALC :")
for r in results_pred[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nTexte ref :")
for r in results_ref[0]:
    print(f"{r['label']}: {r['score']:.4f}")

print("\nDistance euclidienne entre text ori et falc predit")
dist = compute_distance(results_ori, results_pred)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(cosinie_distance(results_ori, results_pred))

print("\nDistance euclidienne entre text ori et falc ref")
dist = compute_distance(results_ori, results_ref)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(cosinie_distance(results_ori, results_ref))

print("\nDistance euclidienne entre falc pred et falc ref")
dist = compute_distance(results_pred, results_ref)
for r in dist:
    print(f"{r}: {dist[r]:.4f}")
print(cosinie_distance(results_pred, results_ref))