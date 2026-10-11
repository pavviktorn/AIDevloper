#!/usr/bin/env python3
import argparse
import base64
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import error, request


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

# DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/Selfie"
DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/Axonlabs_replay_samples"
DEFAULT_API_URL = "https://127.0.0.1:3000/face_liveness_base64"
LABELS = ("real", "fake")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send every image under a folder to receive_face_base64, save the "
            "response beside the image, and compare Analysis result with the "
            "true label from real/fake subfolders."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder to scan for images. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help=f"Endpoint URL. Default: {DEFAULT_API_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images when the output .txt file already exists.",
    )
    return parser.parse_args()


def build_label_stats() -> dict[str, dict[str, int]]:
    return {
        label: {
            "found": 0,
            "processed": 0,
            "evaluated": 0,
            "correct": 0,
            "miss": 0,
            "failed": 0,
        }
        for label in LABELS
    }


def iter_image_files(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            yield path


def get_true_label(image_path: Path) -> Optional[str]:
    for parent in image_path.parents:
        folder_name = parent.name.lower()
        if folder_name == "real":
            return "real"
        elif folder_name == "fake":
            return "fake"
    return None


def get_output_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def dump_text_file(path: Path, content: str) -> None:
    if not content.endswith("\n"):
        content += "\n"
    path.write_text(content, encoding="utf-8")


def response_to_text(response_json: Any) -> str:
    return json.dumps(response_json, indent=2, ensure_ascii=False)


def parse_json_text(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def extract_analysis_from_text(text: str) -> Optional[str]:
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        normalized_key = key.strip().lower().replace(" ", "_")
        if normalized_key in {"analysis_result", "analysisresult", "result"}:
            return value.strip().strip('"').strip("'")
    return None


def extract_analysis_result(response_json: Any) -> Optional[str]:
    candidates = []
    if isinstance(response_json, dict):
        candidates.append(response_json)
        face_liveness = response_json.get("face_liveness")
        if face_liveness is not None:
            candidates.append(face_liveness)

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("Analysis result", "analysis_result", "analysisresult", "result"):
                if key in candidate:
                    value = candidate[key]
                    return str(value).strip() if value is not None else None
        elif isinstance(candidate, str):
            nested_json = parse_json_text(candidate)
            if nested_json is not None:
                nested_result = extract_analysis_result(nested_json)
                if nested_result is not None:
                    return nested_result
            text_result = extract_analysis_from_text(candidate)
            if text_result is not None:
                return text_result

    if isinstance(response_json, str):
        return extract_analysis_from_text(response_json)

    return None


def extract_match_score(response_json: Any) -> Optional[str]:
    candidates = []
    if isinstance(response_json, dict):
        candidates.append(response_json)
        face_liveness = response_json.get("face_liveness")
        if face_liveness is not None:
            candidates.append(face_liveness)

    for candidate in candidates:
        if isinstance(candidate, dict):
            for key in ("Match score", "match_score", "matchscore"):
                if key in candidate:
                    value = candidate[key]
                    return str(value).strip() if value is not None else None
        elif isinstance(candidate, str):
            nested_json = parse_json_text(candidate)
            if nested_json is not None:
                nested_score = extract_match_score(nested_json)
                if nested_score is not None:
                    return nested_score
            for line in candidate.splitlines():
                if ":" not in line:
                    continue
                key, value = line.split(":", 1)
                normalized_key = key.strip().lower().replace(" ", "_")
                if normalized_key in {"match_score", "matchscore"}:
                    return value.strip().strip('"').strip("'")

    if isinstance(response_json, str):
        for line in response_json.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            normalized_key = key.strip().lower().replace(" ", "_")
            if normalized_key in {"match_score", "matchscore"}:
                return value.strip().strip('"').strip("'")

    return None


def normalize_prediction_label(analysis_result: Optional[str]) -> str:
    if analysis_result is not None and analysis_result.strip().lower() == "real":
        return "real"
    return "fake"


def build_request_body(image_path: Path) -> bytes:
    image_bytes = image_path.read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode("ascii")
    payload = {"image_base64": image_base64}
    return json.dumps(payload).encode("utf-8")


def post_image(api_url: str, image_path: Path, timeout: float) -> tuple[int, str, Any]:
    payload = build_request_body(image_path)
    req = request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout) as response:
            status_code = response.getcode()
            body_text = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        status_code = exc.code
        body_text = exc.read().decode("utf-8", errors="replace")
    except error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    response_json = parse_json_text(body_text)
    return status_code, body_text, response_json


def main() -> int:
    args = parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a folder: {input_dir}", file=sys.stderr)
        return 1

    image_paths = list(iter_image_files(input_dir))
    if not image_paths:
        print(f"No image files found under {input_dir}")
        return 0

    label_stats = build_label_stats()
    for image_path in image_paths:
        true_label = get_true_label(image_path)
        if true_label in label_stats:
            label_stats[true_label]["found"] += 1

    total = 0
    correct = 0
    incorrect = 0
    skipped = 0
    failed = 0

    for index, image_path in enumerate(image_paths, start=1):
        true_label = get_true_label(image_path)
        if true_label is None:
            skipped += 1
            print(f"[SKIP] {index}/{len(image_paths)} {image_path} (no real/fake parent folder)")
            continue

        output_path = get_output_path(image_path)
        if args.skip_existing and output_path.exists():
            skipped += 1
            print(f"[SKIP] {index}/{len(image_paths)} {image_path} ({output_path.name} already exists)")
            continue

        total += 1
        label_stats[true_label]["processed"] += 1

        try:
            status_code, body_text, response_json = post_image(args.api_url, image_path, args.timeout)
        except Exception as exc:
            failed += 1
            label_stats[true_label]["failed"] += 1
            error_json = {
                "success": False,
                "error": str(exc),
                "image": str(image_path),
            }
            dump_text_file(output_path, response_to_text(error_json))
            print(f"[FAIL] {index}/{len(image_paths)} {image_path} error={exc}", file=sys.stderr)
            continue

        saved_content = body_text
        if response_json is not None:
            saved_content = response_to_text(response_json)
        dump_text_file(output_path, saved_content)

        analysis_result = extract_analysis_result(response_json if response_json is not None else body_text)
        predicted_label = normalize_prediction_label(analysis_result)
        match_score = extract_match_score(response_json if response_json is not None else body_text)
        response_dict = response_json if isinstance(response_json, dict) else None

        if response_dict is None or status_code != 200 or not response_dict.get("success", False):
            failed += 1
            label_stats[true_label]["failed"] += 1
            message = analysis_result if analysis_result is not None else "N/A"
            shown_match_score = match_score if match_score is not None else "N/A"
            print(
                f"[FAIL] {index}/{len(image_paths)} {image_path} "
                f"status={status_code} analysis={message} match_score={shown_match_score} "
                f"predicted={predicted_label} true={true_label}",
                file=sys.stderr,
            )
            continue

        label_stats[true_label]["evaluated"] += 1
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1
            label_stats[true_label]["correct"] += 1
            state = "OK"
        else:
            incorrect += 1
            label_stats[true_label]["miss"] += 1
            state = "MISS"

        shown_result = analysis_result if analysis_result is not None else "N/A"
        shown_match_score = match_score if match_score is not None else "N/A"
        print(
            f"[{state}] {index}/{len(image_paths)} {image_path} "
            f"true={true_label} analysis={shown_result} match_score={shown_match_score} "
            f"predicted={predicted_label}"
        )

    evaluated = correct + incorrect
    accuracy = (correct / evaluated * 100.0) if evaluated else 0.0

    print("\nSummary")
    print(f"API URL: {args.api_url}")
    print(f"Input folder: {input_dir}")
    print(f"Images found: {len(image_paths)}")
    print(f"Processed: {total}")
    print(f"Evaluated: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Accuracy: {accuracy:.2f}%")
    print("")

    for label in LABELS:
        stats = label_stats[label]
        label_accuracy = (stats["correct"] / stats["evaluated"] * 100.0) if stats["evaluated"] else 0.0
        print(f"{label.capitalize()} count (all found): {stats['found']}")
        print(f"{label.capitalize()} count (processed): {stats['processed']}")
        print(f"{label.capitalize()} count (evaluated): {stats['evaluated']}")
        print(f"{label.capitalize()} accuracy: {label_accuracy:.2f}%")
        print(f"{label.capitalize()} miss: {stats['miss']}")
        print(f"{label.capitalize()} failed: {stats['failed']}")
        print("")

    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
