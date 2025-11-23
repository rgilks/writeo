#!/usr/bin/env python3
"""
Apply corpus-based calibration to the scoring service.
Loads calibration results and generates the calibration function code.

Usage:
    python apply-corpus-calibration.py [calibration-results.json]
"""

import json
import sys
import os

# Default input file
DEFAULT_INPUT_FILE = "calibration-results.json"


def load_calibration_results(input_file: str) -> dict:
    """Load calibration results from JSON file."""
    if not os.path.exists(input_file):
        print(f"ERROR: Calibration results file not found: {input_file}")
        print("Run calibrate-from-corpus.py first to generate calibration data")
        sys.exit(1)

    with open(input_file, "r") as f:
        return json.load(f)


def generate_calibration_function(stats: dict) -> str:
    """
    Generate calibration function code from statistics.
    Uses the same algorithm as calibrate-from-corpus.py to ensure consistency.
    """
    if not stats:
        return "# No calibration data available"

    # Sort CEFR levels by target score mean
    sorted_cefr = sorted(stats.items(), key=lambda x: x[1]["target_score_mean"])

    # Build non-overlapping score ranges
    ranges = []
    for i, (cefr, stat) in enumerate(sorted_cefr):
        if stat["count"] < 3:  # Skip if too few samples
            continue

        # Define range boundaries
        if i == 0:
            score_min = 0.0
        else:
            prev_max = ranges[-1]["score_max"]
            score_min = (prev_max + stat["q25"]) / 2

        if i == len(sorted_cefr) - 1:
            score_max = 9.0
        else:
            next_cefr = sorted_cefr[i + 1][1]
            score_max = (stat["q75"] + next_cefr["q25"]) / 2

        ranges.append(
            {
                "cefr": cefr,
                "stat": stat,
                "score_min": score_min,
                "score_max": score_max,
            }
        )

    # Generate function code
    code_lines = [
        "def calibrate_from_corpus(model_score: float) -> float:",
        '    """',
        "    Calibrate model score based on corpus analysis.",
        "    Maps model scores to appropriate band score ranges using piecewise linear interpolation.",
        "    ",
        f"    Based on analysis of {sum(s['count'] for s in stats.values())} essays with human CEFR labels.",
        '    """',
        "    # Clamp input to valid range",
        "    model_score = max(0.0, min(9.0, model_score))",
        "",
    ]

    # Generate calibration rules with non-overlapping ranges
    for i, range_info in enumerate(ranges):
        cefr = range_info["cefr"]
        stat = range_info["stat"]
        score_min = range_info["score_min"]
        score_max = range_info["score_max"]

        adjustment = stat["adjustment"]
        target_min = stat["target_score_min"]
        target_max = stat["target_score_max"]

        if i == 0:
            code_lines.append(
                f"    # {cefr}: model range [{stat['q25']:.2f}-{stat['q75']:.2f}], target [{target_min:.1f}-{target_max:.1f}]"
            )
            code_lines.append(f"    if model_score <= {score_max:.2f}:")
        elif i == len(ranges) - 1:
            code_lines.append(
                f"    # {cefr}: model range [{stat['q25']:.2f}-{stat['q75']:.2f}], target [{target_min:.1f}-{target_max:.1f}]"
            )
            code_lines.append(f"    elif model_score >= {score_min:.2f}:")
        else:
            code_lines.append(
                f"    # {cefr}: model range [{stat['q25']:.2f}-{stat['q75']:.2f}], target [{target_min:.1f}-{target_max:.1f}]"
            )
            code_lines.append(
                f"    elif {score_min:.2f} <= model_score <= {score_max:.2f}:"
            )

        code_lines.append(
            f"        # Adjust by {adjustment:+.2f} and clamp to {cefr} range"
        )
        code_lines.append(f"        calibrated = model_score + {adjustment:.2f}")
        code_lines.append(
            f"        return max({target_min:.1f}, min({target_max:.1f}, calibrated))"
        )
        code_lines.append("")

    # Default case
    code_lines.append("    # Default: apply general adjustment")
    code_lines.append("    # Model tends to overestimate, so reduce scores")
    code_lines.append("    if model_score > 7.5:")
    code_lines.append("        return model_score - 1.0")
    code_lines.append("    elif model_score > 6.0:")
    code_lines.append("        return model_score - 0.5")
    code_lines.append("    else:")
    code_lines.append("        return max(2.0, model_score)")

    return "\n".join(code_lines)


def main():
    """Main function."""
    input_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT_FILE

    print("=" * 80)
    print("CORPUS-BASED CALIBRATION FUNCTION GENERATOR")
    print("=" * 80)
    print(f"Input file: {input_file}")
    print("=" * 80)

    # Load calibration results
    cal_data = load_calibration_results(input_file)
    stats = cal_data.get("statistics", {})

    if not stats:
        print("ERROR: No statistics found in calibration results file")
        sys.exit(1)

    # Show summary
    total_essays = sum(s["count"] for s in stats.values())
    print(f"\nCalibration based on {total_essays} essays:")
    for cefr in sorted(stats.keys()):
        stat = stats[cefr]
        print(
            f"  {cefr}: {stat['count']} essays, adjustment: {stat['adjustment']:+.2f}"
        )

    # Generate calibration function
    calibration_code = generate_calibration_function(stats)

    # Output
    print("\n" + "=" * 80)
    print("GENERATED CALIBRATION FUNCTION")
    print("=" * 80)
    print(calibration_code)
    print("=" * 80)
    print("\nThis function should replace the current calibration logic in app.py")
    print(
        f"It's based on actual model performance on {total_essays} essays with human CEFR labels."
    )

    # Optionally save to file
    output_file = input_file.replace(".json", "_function.py")
    with open(output_file, "w") as f:
        f.write(calibration_code)
    print(f"\nCalibration function saved to: {output_file}")


if __name__ == "__main__":
    main()
