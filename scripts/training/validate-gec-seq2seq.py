import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from tqdm import tqdm


def load_data(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def generate_corrections(model, tokenizer, texts, device, batch_size=16):
    model.eval()
    predictions = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating corrections"):
        batch = [f"grammar: {t}" for t in texts[i : i + batch_size]]
        inputs = tokenizer(
            batch, return_tensors="pt", max_length=512, truncation=True, padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_length=512, num_beams=4, early_stopping=True
            )

        preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        predictions.extend(preds)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="google/flan-t5-base",
        help="Path to model or huggingface id",
    )
    parser.add_argument(
        "--test_file", default="data/gec-seq2seq/test.jsonl", help="Path to test file"
    )
    parser.add_argument(
        "--output_file", default="predictions.txt", help="Path to save predictions"
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_path).to(device)

    # Load data
    print(f"Loading data from {args.test_file}")
    data = load_data(args.test_file)
    sources = [d["source"] for d in data]

    # Generate
    predictions = generate_corrections(model, tokenizer, sources, device)

    # Save predictions
    with open(args.output_file, "w") as f:
        for p in predictions:
            f.write(p + "\n")

    # Evaluation instructions
    print("Predictions saved. Evaluation skipped in this script.")

    # Since implementing full ERRANT scoring in python script is verbose,
    # we'll suggest using m2scorer or errant_parallel on the output file.
    print(f"Predictions saved to {args.output_file}")
    print("To get official scores, run:")
    print(
        f"errant_parallel -orig <source_file> -cor {args.output_file} -ref <target_m2>"
    )


if __name__ == "__main__":
    main()
