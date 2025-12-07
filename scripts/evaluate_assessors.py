#!/usr/bin/env python3
"""
Evaluate Assessors Script

This script:
1. Loads a sample of essays from scripts/training/data/test.jsonl
2. Sends them to the local API worker (http://localhost:8787)
3. Collects scores from all defined assessors
4. Generates a comparative report
"""

import json
import uuid
import requests
import time
import random
import os
import sys
import numpy as np
from collections import defaultdict

# API Configuration
API_URL = os.environ.get("API_URL", "http://localhost:8787")
API_KEY = os.environ.get("API_KEY", "")  # Load from environment
SUBMISSION_ENDPOINT = f"{API_URL}/v1/text/submissions"
TEST_DATA_PATH = "scripts/training/data/test.jsonl"
SAMPLE_SIZE = 20  # Number of essays to test
TIMEOUT = 120  # Seconds to wait for results


def load_test_data(limit=SAMPLE_SIZE):
    """Load random sample from test.jsonl"""
    try:
        with open(TEST_DATA_PATH, "r") as f:
            lines = f.readlines()

        # Sample random lines if we have enough data
        if len(lines) > limit:
            lines = random.sample(lines, limit)

        data = []
        for line in lines:
            data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def split_input(input_text):
    """
    Split the combined input prompt + essay into question and answer.
    Heuristic: Split by double newline or take last chunk as essay.
    """
    parts = input_text.split("\n\n")
    if len(parts) >= 2:
        question = parts[0].strip()
        answer = "\n\n".join(parts[1:]).strip()
    else:
        # Fallback
        question = "Write an essay."
        answer = input_text
    return question, answer


def submit_essay(question, answer):
    submission_id = str(uuid.uuid4())

    payload = {
        "submissionId": submission_id,
        "submission": [
            {
                "part": 1,
                "answers": [
                    {
                        "id": str(uuid.uuid4()),
                        "questionId": str(uuid.uuid4()),
                        "questionText": question,
                        "text": answer,
                    }
                ],
            }
        ],
        "storeResults": True,
    }

    try:
        # Step 1: POST submission
        res = requests.post(
            SUBMISSION_ENDPOINT,
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Token {API_KEY}",
            },
        )
        if res.status_code not in [200, 201]:
            print(f"  Failed to submit {submission_id}: {res.text}")
            return None, None

        return submission_id, payload
    except Exception as e:
        print(f"  Request error: {e}")
        return None, None


def wait_for_result(submission_id):
    """Poll for results"""
    start_time = time.time()
    url = f"{SUBMISSION_ENDPOINT}/{submission_id}"

    while time.time() - start_time < TIMEOUT:
        try:
            res = requests.get(url, headers={"Authorization": f"Token {API_KEY}"})
            if res.status_code == 200:
                data = res.json()
                status = data.get("status")

                if status == "success":
                    return data
                elif status == "error":
                    print(f"  Assessment failed: {data}")
                    return None

            time.sleep(2)
        except Exception as e:
            print(f"  Polling error: {e}")
            time.sleep(2)

    print(f"  Timeout waiting for {submission_id}")
    return None


def analyze_results(results_data):
    """Analyze collected results against human labels"""
    report = []
    report.append("# Assessor Evaluation Report")
    report.append(f"Sample Size: {len(results_data)} essays")
    report.append("")

    # Init stats buckets
    # assessor_id -> { errors: [], diffs: [] }
    stats = defaultdict(lambda: {"diffs": [], "scores": [], "human_scores": []})

    table_rows = []

    for item in results_data:
        human_score = item["human_score"]
        human_cefr = item["human_cefr"]
        api_res = item["api_result"]

        # safely extract answers
        try:
            answers = api_res["results"]["parts"][0]["answers"][0]
            assessors = answers.get("assessorResults", [])
        except (KeyError, IndexError, TypeError):
            assessors = []

        row = f"| {human_cefr} ({human_score}) |"

        assessor_scores = {}

        for assessor in assessors:
            a_id = assessor["id"]
            # We only care about graders (scores) for quantitative comparison
            if assessor.get("type") == "grader":
                score = assessor.get("overall", 0)
                stats[a_id]["scores"].append(score)
                stats[a_id]["human_scores"].append(human_score)
                stats[a_id]["diffs"].append(score - human_score)
                assessor_scores[a_id] = score

        # Build table row dynamically based on found assessors
        # Just dumping a summary formatted string for now
        scores_str = ", ".join([f"{k}: {v}" for k, v in assessor_scores.items()])
        row += f" {scores_str} |"
        table_rows.append(row)

    report.append("## 1. Individual Essay Performance")
    report.append("| Human Label | Model Scores |")
    report.append("|---|---|")
    report.extend(table_rows)
    report.append("")

    report.append("## 2. Aggregate Performance (MAE & Bias)")
    report.append("MAE = Mean Absolute Error (lower is better)")
    report.append("Bias = Mean Error (positive means overestimation)")
    report.append("")
    report.append("| Assessor | MAE | Bias | Correlation (r) |")
    report.append("|---|---|---|---|")

    for a_id, data in stats.items():
        diffs = np.array(data["diffs"])
        scores = np.array(data["scores"])
        human = np.array(data["human_scores"])

        mae = np.mean(np.abs(diffs))
        bias = np.mean(diffs)
        corr = np.corrcoef(scores, human)[0, 1] if len(scores) > 1 else 0

        report.append(f"| {a_id} | {mae:.2f} | {bias:+.2f} | {corr:.2f} |")

    report.append("")
    report.append("## 3. Recommendations")
    report.append("Based on the data above:")

    # Simple heuristic generation
    best_mae = 999
    best_assessor = None
    for a_id, data in stats.items():
        mae = np.mean(np.abs(data["diffs"]))
        if mae < best_mae:
            best_mae = mae
            best_assessor = a_id

    if best_assessor:
        report.append(
            f"- **Best Performer**: `{best_assessor}` had the lowest error ({best_mae:.2f})."
        )

    overestimators = [
        a_id for a_id, data in stats.items() if np.mean(data["diffs"]) > 0.5
    ]
    if overestimators:
        report.append(
            f"- **Overestimation**: The following models tend to score high: {', '.join(overestimators)}."
        )

    return "\n".join(report)


def main():
    print(f"Loading data from {TEST_DATA_PATH}...")
    data = load_test_data()
    print(f"Loaded {len(data)} essays.")

    results_collection = []

    print("Starting evaluation...")
    for i, entry in enumerate(data):
        input_text = entry["input"]
        human_score = entry["target"]
        human_cefr = entry["cefr"]

        question, answer = split_input(input_text)

        print(f"[{i + 1}/{len(data)}] Submitting essay ({human_cefr})...")
        sub_id, _ = submit_essay(question, answer)

        if sub_id:
            res = wait_for_result(sub_id)
            if res:
                results_collection.append(
                    {
                        "human_score": human_score,
                        "human_cefr": human_cefr,
                        "api_result": res,
                        "original_entry": entry,
                    }
                )
        else:
            print("  Skipping due to submission failure.")

        time.sleep(0.5)

    if not results_collection:
        print("No results collected.")
        return

    print("\nGenerating report...")
    report = analyze_results(results_collection)

    with open("assessor_report.md", "w") as f:
        f.write(report)

    print("Report saved to assessor_report.md")
    print("\n" + report)


if __name__ == "__main__":
    main()
