from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "model/falc_dpo_model"

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_path)




def reward_function(completions, prompts, **kwargs):
    rewards = []

    originals = kwargs["original"]
    references = kwargs["reference"]

    for completion, original, reference in zip(
        completions,
        originals,
        references
    ):

        generated = completion[0]["content"]

        sari = compute_sari(original, generated, reference)
        bert = compute_bert(reference, generated)
        emotion = compute_emotion_similarity(original, generated)
        fre = compute_fre(generated)

        reward = (
            0.25 * sari +
            0.30 * bert +
            0.20 * fre +
            0.25 * emotion
        )

        rewards.append(reward)

    return rewards





from trl import GRPOConfig

training_args = GRPOConfig(
    output_dir="grpo_model",
    learning_rate=1e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=10,
    max_prompt_length=1024,
    max_completion_length=256,
    num_generations=4,
)





from trl import GRPOTrainer

trainer = GRPOTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    reward_funcs=reward_function,
)




trainer.train()

trainer.save_model("model/em_phrase_grpo_model")