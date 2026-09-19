import json
from ollama import chat

data = []

with open('data/val.json', 'r') as file:
    for line in file:
        if line.strip(): 
            data.append(json.loads(line))

sentences = [i['original'] for i in data]

sentences = sentences[:2]

from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "model/em_phrase_dpo_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

results = []

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
- Identifie l'émotion principale de chaque phrase : joie, tristesse, peur, colère, surprise ou neutre.
- Identifie aussi l'intensité de cette émotion : faible, moyenne ou forte.
- Garde cette émotion dans la phrase simplifiée.
- Utilise des mots émotionnels simples pour montrer cette émotion.
- Fais ressentir clairement l'émotion dans chaque phrase importante.

Méthode obligatoire :
1. Lire le texte entier attentivement.
2. Identifier le sujet principal.
3. Identifier l'émotion principale de la phrase et son intensité.
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



for idx, sentence in enumerate(sentences):
    print(idx)
    sentence_entry = {
        "sentence_id": idx + 1,
        "original_text": sentence,
        "falc":data[idx]['falc'],
        "runs": []
    }

    messages = [
        {
            "role": "system",
            "content": build_system_prompt()
        },
        {
            "role": "user",
            "content": sentence
        }
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        # sentence,
        tokenize=False,
        add_generation_prompt=True
    )
    
    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    ).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False
    )
    generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
    
    response = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    )

    
    sentence_entry["runs"].append({
        "falc": response
    })
        
    results.append(sentence_entry)
    
print(results)
# with open("results/falc_dpo_em_phrase.json", "w", encoding="utf-8") as f:
#     json.dump(results, f, indent=4)