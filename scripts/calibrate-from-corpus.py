#!/usr/bin/env python3
"""
Calibration script using corpus data.
Scores essays with human CEFR labels and builds calibration mapping.

Usage:
    CORPUS_PATH=/path/to/corpus.tsv MODAL_URL=https://... python calibrate-from-corpus.py
"""

import pandas as pd
import requests
import uuid
import json
import numpy as np
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import time

# Configuration from environment variables
MODAL_URL = os.getenv("MODAL_URL", os.getenv("MODAL_GRADE_URL", ""))
CORPUS_PATH = os.getenv("CORPUS_PATH", "")
MAX_ESSAYS = int(os.getenv("MAX_ESSAYS", "100"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", "calibration-results.json")

# CEFR to band score mapping (for reference)
CEFR_TO_SCORE = {
    "A2": (2.0, 4.0),
    "A2+": (3.0, 4.5),
    "B1": (4.0, 5.0),
    "B1+": (4.5, 5.5),
    "B2": (5.5, 6.5),
    "B2+": (6.5, 7.0),
    "C1": (7.0, 8.5),
    "C1+": (8.0, 9.0),
    "C2": (8.5, 9.0),
}


def score_essay(question: str, answer: str) -> Dict:
    """Score an essay using the Modal API."""
    submission_id = str(uuid.uuid4())
    
    request_data = {
        "submission_id": submission_id,
        "template": {"name": "essay-task-2", "version": 1},
        "parts": [
            {
                "part": 1,
                "answers": [
                    {
                        "id": str(uuid.uuid4()),
                        "question_id": str(uuid.uuid4()),
                        "question_text": question or "Write an essay.",
                        "answer_text": answer
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{MODAL_URL}/grade",
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def extract_score(result: Dict) -> Tuple[Optional[float], Optional[str]]:
    """Extract score from API response."""
    if "error" in result:
        return None, None
    
    if result.get("status") != "success":
        return None, None
    
    try:
        assessor_result = result["results"]["parts"][0]["assessor-results"][0]
        overall = assessor_result["overall"]
        label = assessor_result["label"]
        return overall, label
    except (KeyError, IndexError):
        return None, None


def load_corpus_sample(corpus_path: str, max_essays: int = 500) -> pd.DataFrame:
    """Load a sample of essays with human CEFR labels."""
    if not os.path.exists(corpus_path):
        print(f"ERROR: Corpus file not found: {corpus_path}")
        print("Set CORPUS_PATH environment variable or update default path")
        sys.exit(1)
    
    print(f"Loading corpus from {corpus_path}...")
    df = pd.read_csv(corpus_path, sep="\t", low_memory=False)
    
    # Filter for final versions with human CEFR labels
    df = df[df["is_final_version"] == True]
    df = df[df["humannotator_cefr_level"].notna()]
    df = df[df["humannotator_cefr_level"] != "NA"]
    df = df[df["text"].notna()]
    df = df[df["text"].str.len() > 50]  # Minimum length
    
    print(f"Found {len(df)} essays with human CEFR labels")
    
    # Sample evenly across CEFR levels
    cefr_counts = df["humannotator_cefr_level"].value_counts()
    print(f"CEFR distribution: {cefr_counts.to_dict()}")
    
    # Sample up to max_essays, trying to balance across levels
    sampled = []
    per_level = max(10, max_essays // len(cefr_counts))
    
    for cefr in cefr_counts.index:
        level_df = df[df["humannotator_cefr_level"] == cefr]
        sample_size = min(per_level, len(level_df))
        sampled.append(level_df.sample(n=sample_size, random_state=42))
    
    result_df = pd.concat(sampled, ignore_index=True)
    result_df = result_df.sample(n=min(max_essays, len(result_df)), random_state=42)
    
    print(f"Sampled {len(result_df)} essays for calibration")
    return result_df


def build_calibration_mapping(results: List[Dict]) -> Dict:
    """Build calibration mapping from model scores to target band scores."""
    # Group by human CEFR level
    by_cefr = defaultdict(list)
    for r in results:
        if r["model_score"] is not None and r["human_cefr"]:
            by_cefr[r["human_cefr"]].append(r["model_score"])
    
    # Calculate statistics per CEFR level
    calibration_stats = {}
    for cefr, scores in by_cefr.items():
        if len(scores) > 0:
            target_min, target_max = CEFR_TO_SCORE.get(cefr, (0, 9))
            target_mean = (target_min + target_max) / 2
            
            calibration_stats[cefr] = {
                "count": len(scores),
                "mean": float(np.mean(scores)),
                "median": float(np.median(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "q25": float(np.percentile(scores, 25)),
                "q75": float(np.percentile(scores, 75)),
                "target_score_min": target_min,
                "target_score_max": target_max,
                "target_score_mean": target_mean,
                "adjustment": target_mean - float(np.mean(scores)),
            }
    
    return calibration_stats


def generate_calibration_function(stats: Dict) -> str:
    """
    Generate Python code for calibration function with non-overlapping ranges.
    Uses piecewise linear interpolation for smooth calibration.
    """
    if not stats:
        return "# No calibration data available"
    
    # Sort CEFR levels by target score mean
    sorted_cefr = sorted(stats.items(), key=lambda x: x[1]["target_score_mean"])
    
    # Build non-overlapping score ranges
    # Use percentiles to define ranges, ensuring no overlap
    ranges = []
    for i, (cefr, stat) in enumerate(sorted_cefr):
        if stat["count"] < 3:  # Skip if too few samples
            continue
        
        # Define range boundaries
        if i == 0:
            # First range: start from 0
            score_min = 0.0
        else:
            # Start from midpoint between previous and current
            prev_max = ranges[-1]["score_max"]
            score_min = (prev_max + stat["q25"]) / 2
        
        if i == len(sorted_cefr) - 1:
            # Last range: extend to 9.0
            score_max = 9.0
        else:
            # End at midpoint between current and next
            next_cefr = sorted_cefr[i + 1][1]
            score_max = (stat["q75"] + next_cefr["q25"]) / 2
        
        ranges.append({
            "cefr": cefr,
            "stat": stat,
            "score_min": score_min,
            "score_max": score_max,
        })
    
    # Generate function code
    code_lines = [
        'def calibrate_from_corpus(model_score: float) -> float:',
        '    """',
        '    Calibrate model score based on corpus analysis.',
        '    Maps model scores to appropriate band score ranges using piecewise linear interpolation.',
        '    ',
        f'    Based on analysis of {sum(s["count"] for s in stats.values())} essays with human CEFR labels.',
        '    """',
        '    # Clamp input to valid range',
        '    model_score = max(0.0, min(9.0, model_score))',
        '',
    ]
    
    # Generate calibration rules with non-overlapping ranges
    for i, range_info in enumerate(ranges):
        cefr = range_info["cefr"]
        stat = range_info["stat"]
        score_min = range_info["score_min"]
        score_max = range_info["score_max"]
        
        # Calculate linear transformation: model_score -> target_score
        # Use mean adjustment for simplicity
        adjustment = stat["adjustment"]
        target_min = stat["target_score_min"]
        target_max = stat["target_score_max"]
        
        if i == 0:
            code_lines.append(f'    # {cefr}: model range [{stat["q25"]:.2f}-{stat["q75"]:.2f}], target [{target_min:.1f}-{target_max:.1f}]')
            code_lines.append(f'    if model_score <= {score_max:.2f}:')
        elif i == len(ranges) - 1:
            code_lines.append(f'    # {cefr}: model range [{stat["q25"]:.2f}-{stat["q75"]:.2f}], target [{target_min:.1f}-{target_max:.1f}]')
            code_lines.append(f'    elif model_score >= {score_min:.2f}:')
        else:
            code_lines.append(f'    # {cefr}: model range [{stat["q25"]:.2f}-{stat["q75"]:.2f}], target [{target_min:.1f}-{target_max:.1f}]')
            code_lines.append(f'    elif {score_min:.2f} <= model_score <= {score_max:.2f}:')
        
        # Apply linear adjustment and clamp to target range
        code_lines.append(f'        # Adjust by {adjustment:+.2f} and clamp to {cefr} range')
        code_lines.append(f'        calibrated = model_score + {adjustment:.2f}')
        code_lines.append(f'        return max({target_min:.1f}, min({target_max:.1f}, calibrated))')
        code_lines.append('')
    
    # Default case (shouldn't be reached, but safety)
    code_lines.append('    # Default: apply general adjustment')
    code_lines.append('    # Model tends to overestimate, so reduce scores')
    code_lines.append('    if model_score > 7.5:')
    code_lines.append('        return model_score - 1.0')
    code_lines.append('    elif model_score > 6.0:')
    code_lines.append('        return model_score - 0.5')
    code_lines.append('    else:')
    code_lines.append('        return max(2.0, model_score)')
    
    return '\n'.join(code_lines)


def main():
    """Main calibration process."""
    if not MODAL_URL:
        print("ERROR: MODAL_URL or MODAL_GRADE_URL environment variable is required")
        print("Set it via: export MODAL_URL=https://your-username--writeo-essay-fastapi-app.modal.run")
        sys.exit(1)
    
    if not CORPUS_PATH:
        print("ERROR: CORPUS_PATH environment variable is required")
        print("Set it via: export CORPUS_PATH=/path/to/corpus.tsv")
        sys.exit(1)
    
    print("="*80)
    print("CALIBRATION FROM CORPUS DATA")
    print("="*80)
    print(f"Modal URL: {MODAL_URL}")
    print(f"Corpus path: {CORPUS_PATH}")
    print(f"Max essays: {MAX_ESSAYS}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*80)
    
    # Load corpus sample
    df = load_corpus_sample(CORPUS_PATH, MAX_ESSAYS)
    
    # Score essays
    results = []
    print(f"\nScoring {len(df)} essays...")
    for idx, row in df.iterrows():
        essay_text = row["text"]
        human_cefr = row["humannotator_cefr_level"]
        prompt_id = row.get("public_prompt_id", "unknown")
        
        prompt_text = f"Essay prompt {prompt_id}"
        
        if (idx + 1) % 10 == 0:
            print(f"[{idx+1}/{len(df)}] Scoring essay (Human CEFR: {human_cefr})...")
        
        result = score_essay(prompt_text, essay_text)
        model_score, model_cefr = extract_score(result)
        
        if model_score is not None:
            results.append({
                "human_cefr": human_cefr,
                "model_score": model_score,
                "model_cefr": model_cefr,
                "essay_length": len(essay_text),
            })
        else:
            print(f"  Error scoring essay {idx+1}: {result.get('error', 'Unknown error')}")
        
        # Rate limiting
        time.sleep(1)
    
    if not results:
        print("ERROR: No essays were successfully scored!")
        sys.exit(1)
    
    # Build calibration mapping
    print("\n" + "="*80)
    print("CALIBRATION STATISTICS")
    print("="*80)
    
    stats = build_calibration_mapping(results)
    
    for cefr in sorted(stats.keys()):
        stat = stats[cefr]
        print(f"\n{cefr}:")
        print(f"  Count: {stat['count']}")
        print(f"  Model Score: {stat['mean']:.2f} ± {stat['std']:.2f} (range: {stat['min']:.2f}-{stat['max']:.2f})")
        print(f"  Target Score: {stat['target_score_min']:.1f}-{stat['target_score_max']:.1f} (mean: {stat['target_score_mean']:.2f})")
        print(f"  Adjustment needed: {stat['adjustment']:+.2f}")
    
    # Generate calibration code
    calibration_code = generate_calibration_function(stats)
    
    # Save results
    with open(OUTPUT_FILE, "w") as f:
        json.dump({
            "statistics": stats,
            "results": results,
            "calibration_code": calibration_code,
            "config": {
                "modal_url": MODAL_URL,
                "corpus_path": CORPUS_PATH,
                "max_essays": MAX_ESSAYS,
                "total_scored": len(results),
            }
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"Calibration function generated (see {OUTPUT_FILE})")
    print("="*80)
    
    # Print calibration function for quick review
    print("\nGenerated calibration function:")
    print("-"*80)
    print(calibration_code)
    print("-"*80)


if __name__ == "__main__":
    main()
