import json
from ollama import chat

data = []

with open('../data/train.json', 'r') as file:
    for line in file:
        if line.strip(): 
            data.append(json.loads(line))

sentences = [i['original'] for i in data]

# sentences = sentences[0:15]

models = ["mistral:latest","llama3:latest", "gemma4:latest"]#, "ministral-3:3b", "qwen2.5:3b"]

configs = [
    {"temperature": 0.3, "num_ctx": 12288},
    {"temperature": 0.5, "num_ctx": 12288},
    {"temperature": 0.7, "num_ctx": 12288},
    {"temperature": 0.9, "num_ctx": 12288},
]

results = []

examples = []

with open('../data/test.json', 'r') as file:
    for line in file:
        if line.strip(): 
            examples.append(json.loads(line))

def build_system_prompt():
    return f"""
Tu es un expert du FALC (Facile à Lire et à Comprendre) et de la simplification émotionnelle des textes. Ton objectif est de transformer un texte complexe en un texte FALC clair, accessible et émotionnellement fidèle. Le texte final doit être facile à comprendre pour les personnes ayant des difficultés de compréhension (handicap mental, dyslexie, personnes âgées, apprenants du français, etc.) tout en gardant l'émotion du texte original très présente et naturelle.

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

Règles émotionnelles très importantes :
- Identifie l'émotion principale du texte : joie, tristesse, peur, colère, surprise ou neutre.
- Identifie aussi l'intensité de cette émotion : faible, moyenne ou forte.
- Garde cette émotion tout au long du texte simplifié.
- Utiliser des mots émotionnels simples pour montrer cette émotion.
- Fais ressentir clairement l'émotion dans chaque phrase importante.

Méthode obligatoire :
1. Lire le texte entier attentivement.
2. Identifier le sujet principal.
3. Identifier l'émotion principale et son intensité.
4. Choisir des mots simples qui transmettent cette émotion.
5. Réécrire le texte en FALC en gardant les émotions visibles et naturelles.

Consignes de sortie : 
- Réponds uniquement avec le texte réécrit en FALC. 
- Ne donne aucune explication. 
- N'ajoute pas de titre. 
- N'ajoute pas de section. 
- N'ajoute pas de commentaire. 
- Ne fais pas de liste. 
- Garde le même contexte et le même message que le texte original. 
- Conserve les émotions de manière claire et naturelle.
"""

few_shot_messages = [
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[0]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[0]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[1]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[1]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[2]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[2]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[3]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[3]['falc']}"
    },
    {
        "role": "user",
        "content": f"Réécris ce texte en FALC : {examples[4]['original']}"  
    },
    {
        "role": "assistant",
        "content": f"{examples[4]['falc']}"
    },
]

# model = models[0]

for model in models: 
    for idx, sentence in enumerate(sentences):
        print(idx)
        sentence_entry = {
            "sentence_id": idx + 1,
            "model": model,
            "original_text": sentence,
            "falc":data[idx]['falc'],
            "runs": []
        }
        
        for config in configs:
            print(config)
            response = chat(
                model = model,
                messages = [
                    {
                        'role': 'system',
                        'content': build_system_prompt(),
                    },
                    *few_shot_messages,
                    {
                        'role': 'user',
                        'content': f"{sentence}",
                    },
                ],
                options = config
            )

            falc = response['message']['content']

            sentence_entry["runs"].append({
                "parameters": config,
                "summary": falc
            })
            
        results.append(sentence_entry)

with open("../results/falc_generes_2.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)