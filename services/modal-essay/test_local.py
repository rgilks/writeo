#!/usr/bin/env python3
"""Local testing script for modal-essay service.

This script tests the essay scoring service locally before deployment.
It can run the FastAPI server locally or test against a deployed Modal endpoint.
"""

import argparse
import json
import os
import sys
import traceback
from typing import Any
from uuid import uuid4

import requests  # type: ignore[import-untyped]
import uvicorn

# Add the current directory to the path so we can import the modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api import create_fastapi_app
from api.handlers_submission import process_submission
from schemas import ModalAnswer, ModalPart, ModalRequest


def create_test_request() -> ModalRequest:
    """Create a test request with sample essay data."""
    answer_text = (
        "Last weekend was absolutely fantastic! I had the opportunity to visit my grandparents "
        "who live in a beautiful countryside village. On Saturday morning, I woke up early and "
        "drove for about two hours through scenic countryside roads. The journey itself was "
        "delightful, with rolling hills and green fields stretching as far as the eye could see.\n\n"
        "When I arrived, my grandmother had prepared a delicious homemade lunch with all my "
        "favorite dishes. We spent the afternoon catching up on family news and looking through "
        "old photo albums. My grandfather showed me his vegetable garden, which was thriving "
        "with tomatoes, cucumbers, and various herbs. I helped him water the plants and learned "
        "about different gardening techniques.\n\n"
        "In the evening, we all went for a peaceful walk along a nearby river. The sunset was "
        "breathtaking, painting the sky in shades of orange and pink. We returned home and spent "
        "the rest of the evening playing board games and sharing stories. It was a perfect day "
        "filled with love, laughter, and wonderful memories.\n\n"
        "On Sunday, I helped my grandparents with some household chores before heading back home. "
        "The weekend reminded me of the importance of spending quality time with family and "
        "appreciating the simple pleasures in life."
    )

    return ModalRequest(
        submission_id=str(uuid4()),
        template={"name": "essay-task-2", "version": 1},
        parts=[
            ModalPart(
                part=1,
                answers=[
                    ModalAnswer(
                        id=str(uuid4()),
                        question_id=str(uuid4()),
                        question_text="Describe your weekend. What did you do?",
                        answer_text=answer_text,
                    )
                ],
            )
        ],
    )


def print_results(result: dict[str, Any], show_full_response: bool = False) -> None:
    """Print assessment results in a formatted way."""
    print("✅ Processing completed successfully!")
    print("\n📊 Results:")
    print(f"  Status: {result.get('status')}")

    if result.get("results") and result["results"].get("parts"):
        for part in result["results"]["parts"]:
            print(f"\n  Part {part.get('part')}:")
            for answer in part.get("answers", []):
                print(f"    Answer ID: {answer.get('id')}")
                assessor_results = answer.get("assessorResults", [])
                for ar in assessor_results:
                    print(f"      Assessor: {ar.get('id')} ({ar.get('name')})")
                    print(f"        Overall: {ar.get('overall')} ({ar.get('label')})")
                    dimensions = ar.get("dimensions", {})
                    if dimensions:
                        print("        Dimensions:")
                        for dim, score in dimensions.items():
                            print(f"          {dim}: {score}")
    else:
        print("  ⚠️ No results found!")
        if show_full_response:
            print(f"  Full response: {json.dumps(result, indent=2)}")


def test_local_processing(model_key: str = "engessay") -> dict[str, Any]:
    """Test the scoring logic locally without running a server."""
    print(f"\n{'=' * 60}")
    print(f"🧪 Testing local processing with model: {model_key}")
    print(f"{'=' * 60}\n")

    request = create_test_request()

    try:
        result = process_submission(request, model_key)
        result_dict: dict[str, Any] = result.model_dump(
            mode="json", exclude_none=True, by_alias=True
        )
        print_results(result_dict)
        return result_dict
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def test_remote_endpoint(url: str, api_key: str, model_key: str = "engessay") -> dict[str, Any]:
    """Test against a deployed Modal endpoint."""
    print(f"\n{'=' * 60}")
    print(f"🌐 Testing remote endpoint: {url}")
    print(f"   Model: {model_key}")
    print(f"{'=' * 60}\n")

    request = create_test_request()
    request_dict = request.model_dump(mode="json", exclude_none=True, by_alias=True)

    headers = {"Content-Type": "application/json", "Authorization": f"Token {api_key}"}
    params = {"model_key": model_key} if model_key else {}

    try:
        response = requests.post(
            f"{url}/grade", json=request_dict, headers=headers, params=params, timeout=60
        )

        print(f"📡 Response Status: {response.status_code}")

        if response.status_code == 200:
            result: dict[str, Any] = response.json()
            print_results(result, show_full_response=True)
            return result
        else:
            print(f"❌ Request failed with status {response.status_code}")
            print(f"   Response: {response.text}")
            return {
                "error": f"HTTP {response.status_code}",
                "response": response.text,
            }

    except Exception as e:
        print(f"❌ Error during request: {e}")
        traceback.print_exc()
        return {"error": str(e)}


def run_local_server(port: int = 8000) -> None:
    """Run the FastAPI server locally using uvicorn."""
    app = create_fastapi_app()

    print(f"\n{'=' * 60}")
    print(f"🚀 Starting local FastAPI server on port {port}")
    print(f"   Swagger UI: http://localhost:{port}/docs")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"{'=' * 60}\n")

    # Set a default API key for local testing if not set
    if not os.getenv("MODAL_API_KEY"):
        os.environ["MODAL_API_KEY"] = "test-key-local-dev"
        print("⚠️  MODAL_API_KEY not set, using 'test-key-local-dev' for local testing")
        print("   Set MODAL_API_KEY environment variable to use a real key\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")


def main() -> None:
    """Main test function."""
    parser = argparse.ArgumentParser(description="Test modal-essay service locally or remotely")
    parser.add_argument(
        "--mode",
        choices=["local", "remote", "server"],
        default="local",
        help="Test mode: local (direct function call), remote (HTTP request), or server (run local server)",
    )
    parser.add_argument("--url", help="Remote endpoint URL (required for remote mode)")
    parser.add_argument("--api-key", help="API key for remote endpoint (required for remote mode)")
    parser.add_argument("--model", default="engessay", help="Model key to use (default: engessay)")
    parser.add_argument(
        "--port", type=int, default=8000, help="Port for local server (default: 8000)"
    )

    args = parser.parse_args()

    if args.mode == "server":
        run_local_server(args.port)
    elif args.mode == "remote":
        if not args.url:
            print("❌ Error: --url is required for remote mode")
            sys.exit(1)
        if not args.api_key:
            print("❌ Error: --api-key is required for remote mode")
            sys.exit(1)
        test_remote_endpoint(args.url, args.api_key, args.model)
    else:  # local
        test_local_processing(args.model)


if __name__ == "__main__":
    main()
