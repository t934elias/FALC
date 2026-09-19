from transformers import AutoModelForCausalLM, AutoTokenizer
from copy import deepcopy
from datasets import load_dataset
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
import torch

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

model_name = "mistralai/Mistral-7B-Instruct-v0.3"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # torch_dtype="auto",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ref_model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype="auto",
#     device_map="auto"
# )

def format_example(example):
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": example["prompt"]}
    ]

    example["prompt"] = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return example

dataset = load_dataset(
    "json",
    data_files="results/dpo_dataset.jsonl"
)

dataset = dataset.map(format_example)

# print(dataset["train"][0])
# print(len(dataset["train"]))

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    # target_modules="all-linear",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM"
)


training_args = DPOConfig(
    output_dir="mistral-falc-dpo",
    learning_rate=5e-6,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=500,
    beta=0.1, # preference strength
    bf16=True,
    remove_unused_columns=False
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    peft_config=peft_config
)

trainer.train()
trainer.save_model("model/falc_dpo_model")
tokenizer.save_pretrained("model/falc_dpo_model")