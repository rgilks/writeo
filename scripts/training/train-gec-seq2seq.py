import modal

# Modal setup
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "transformers==4.36.0",
        "accelerate==0.25.0",
        "datasets==2.15.0",
        "scikit-learn==1.3.2",
        "numpy==1.26.2",
        "sentencepiece>=0.1.99",  # Required for T5
    )
    # Mount data directory
    .add_local_dir("data/gec-seq2seq", remote_path="/data")
    # Mount scripts directory (for imports if needed, mostly for self)
    .add_local_dir("scripts/training", remote_path="/scripts/training")
)

app = modal.App("writeo-gec-training", image=image)

# Volume for model checkpoints
volume = modal.Volume.from_name("writeo-gec-models", create_if_missing=True)


@app.function(
    gpu="A10G",  # NVIDIA A10G (24GB VRAM)
    timeout=7200,  # 2 hours
    volumes={"/checkpoints": volume},
)
def train_gec_model():
    """Train GEC Seq2Seq model on Modal."""
    import os

    import datasets
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
        set_seed,
    )

    print("=" * 80)
    print("GEC SEQ2SEQ TRAINING (FLAN-T5-BASE)")
    print("=" * 80)

    # Hardcoded arguments for Modal run
    # (In a real CLI we might pass these via function args, but this is simpler)
    model_name = "google/flan-t5-base"
    train_file = "/data/train.jsonl"
    val_file = "/data/dev.jsonl"
    output_dir = "/checkpoints/gec-seq2seq-v1"

    # Training Config
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=4,  # Effective batch size = 32
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=10,
        predict_with_generate=True,
        bf16=True,  # Use BF16 for T5 stability on A10G
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=["none"],  # Disable wandb/tensorboard for now
        generation_max_length=512,
        seed=42,
    )

    print(f"Model: {model_name}")
    print(f"Train file: {train_file}")
    print(f"Output dir: {output_dir}")

    # Set seed
    set_seed(training_args.seed)

    # Load datasets
    data_files = {}
    if os.path.exists(train_file):
        data_files["train"] = train_file
    else:
        print(f"Error: Train file not found at {train_file}")
        # List dir to debug
        print(f"Listing /data: {os.listdir('/data')}")
        return

    if os.path.exists(val_file):
        data_files["validation"] = val_file

    raw_datasets = datasets.load_dataset("json", data_files=data_files)
    print(f"Loaded datasets: {raw_datasets}")

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Preprocessing
    prefix = "grammar: "
    max_source_length = 512
    max_target_length = 512

    def preprocess_function(examples):
        inputs = [prefix + doc for doc in examples["source"]]
        targets = [doc for doc in examples["target"]]

        model_inputs = tokenizer(inputs, max_length=max_source_length, truncation=True)

        # Use text_target as per warning
        labels = tokenizer(
            text_target=targets, max_length=max_target_length, truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    if "train" in raw_datasets:
        train_dataset = raw_datasets["train"]
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=raw_datasets["train"].column_names,
            desc="Running tokenizer on train dataset",
        )
    else:
        train_dataset = None

    if "validation" in raw_datasets:
        eval_dataset = raw_datasets["validation"]
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=4,
            remove_columns=raw_datasets["validation"].column_names,
            desc="Running tokenizer on validation dataset",
        )
    else:
        eval_dataset = None

    # Data collator
    label_pad_token_id = -100
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if train_dataset:
        print("Starting training...")
        train_result = trainer.train()
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        # Persist to volume (trainer.save_model saves to output_dir which is in /checkpoints volume)
        volume.commit()
        print("Training complete and model saved to volume.")

    # Evaluation
    if eval_dataset:
        print("Evaluating...")
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    return "Done"


@app.local_entrypoint()
def main():
    """Run training."""
    print("Submitting training job to Modal...")
    result = train_gec_model.remote()
    print("Job finished:", result)


if __name__ == "__main__":
    # If run locally, this will trigger the local_entrypoint which runs remote
    main()
