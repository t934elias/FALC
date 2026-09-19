from transformers import AutoModelForCausalLM, AutoTokenizer
import json

model_path = "model/falc_dpo_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype="auto"
)

configs = [
    {"temperature": 0.3, "num_ctx": 12288},
    {"temperature": 0.5, "num_ctx": 12288},
    {"temperature": 0.7, "num_ctx": 12288},
    {"temperature": 0.9, "num_ctx": 12288},
]

data = []

with open('data/val.json', 'r') as file:
    for line in file:
        if line.strip(): 
            data.append(json.loads(line))

sentences = [i['original'] for i in data]
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

for idx, sentence in enumerate(sentences):
    print(idx)
    sentence_entry = {
        "sentence_id": idx + 1,
        "original_text": sentence,
        "falc_humain":data[idx]['falc'],
        "runs": []
    }

    for config in configs:
        print(config)

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
        do_sample=True,
        temperature=config['temperature'],
        top_p=0.9
        )
        generated_tokens = outputs[0][inputs.input_ids.shape[1]:]   

        falc = tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )
        sentence_entry["runs"].append({
                    "falc": falc
                })
            
    results.append(sentence_entry)

with open("results/falc_generes_dpo.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)