#!/usr/bin/env python3
import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterable
from urllib import error, request

from test_folder import (
    DEFAULT_INPUT_DIR,
    LABELS,
    build_label_stats,
    dump_text_file,
    extract_analysis_result,
    extract_match_score,
    get_output_path,
    get_true_label,
    iter_image_files,
    normalize_prediction_label,
    parse_json_text,
    response_to_text,
)


DEFAULT_BATCH_API_URL = "http://127.0.0.1:5000/face_liveness_base64_batch"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send images under a folder to the batched receive_face_base64 endpoint, "
            "save each response beside the image, and compare Analysis result with "
            "the true label from real/fake subfolders."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder to scan for images. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_BATCH_API_URL,
        help=f"Batch endpoint URL. Default: {DEFAULT_BATCH_API_URL}",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Per-request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=24,
        help="Number of images per batch request. Default: 4",
    )
    parser.add_argument(
        "--workers",
        type=int,
        dest="batch_size",
        help="Alias for --batch-size.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images when the output .txt file already exists.",
    )
    return parser.parse_args()


def chunked(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start:start + chunk_size]


def build_batch_request_body(image_paths: list[Path]) -> bytes:
    images_base64 = []
    for image_path in image_paths:
        image_bytes = image_path.read_bytes()
        images_base64.append(base64.b64encode(image_bytes).decode("ascii"))
    payload = {"images_base64": images_base64}
    return json.dumps(payload).encode("utf-8")


def post_batch(api_url: str, image_paths: list[Path], timeout: float) -> tuple[int, str, Any]:
    payload = build_batch_request_body(image_paths)
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


def save_failure_result(output_path: Path, image_path: Path, message: str) -> None:
    error_json = {
        "success": False,
        "error": message,
        "image": str(image_path),
    }
    dump_text_file(output_path, response_to_text(error_json))


def handle_batch_results(
    batch: list[dict[str, Any]],
    results: list[Any],
    image_count: int,
    label_stats: dict[str, dict[str, int]],
) -> tuple[int, int, int]:
    correct = 0
    incorrect = 0
    failed = 0

    for item, result in zip(batch, results):
        if isinstance(result, dict):
            dump_text_file(item["output_path"], response_to_text(result))
        else:
            dump_text_file(item["output_path"], json.dumps(result, ensure_ascii=False, indent=2))

        analysis_result = extract_analysis_result(result)
        match_score = extract_match_score(result)
        predicted_label = normalize_prediction_label(analysis_result)

        if not isinstance(result, dict) or not result.get("success", False):
            failed += 1
            label_stats[item["true_label"]]["failed"] += 1
            error_message = "N/A"
            if isinstance(result, dict):
                error_message = str(result.get("error", "N/A"))
            shown_result = analysis_result if analysis_result is not None else "N/A"
            shown_match_score = match_score if match_score is not None else "N/A"
            print(
                f"[FAIL] {item['index']}/{image_count} {item['image_path']} "
                f"error={error_message} analysis={shown_result} "
                f"match_score={shown_match_score} predicted={predicted_label} "
                f"true={item['true_label']}",
                file=sys.stderr,
            )
            continue

        label_stats[item["true_label"]]["evaluated"] += 1
        if predicted_label == item["true_label"]:
            correct += 1
            label_stats[item["true_label"]]["correct"] += 1
            state = "OK"
        else:
            incorrect += 1
            label_stats[item["true_label"]]["miss"] += 1
            state = "MISS"

        shown_result = analysis_result if analysis_result is not None else "N/A"
        shown_match_score = match_score if match_score is not None else "N/A"
        print(
            f"[{state}] {item['index']}/{image_count} {item['image_path']} "
            f"true={item['true_label']} analysis={shown_result} "
            f"match_score={shown_match_score} predicted={predicted_label}"
        )

    return correct, incorrect, failed


def process_batch_with_retry(
    batch: list[dict[str, Any]],
    image_count: int,
    api_url: str,
    timeout: float,
    label_stats: dict[str, dict[str, int]],
) -> tuple[int, int, int, int]:
    batch_image_paths = [item["image_path"] for item in batch]
    batch_t0 = time.time()

    try:
        status_code, body_text, response_json = post_batch(api_url, batch_image_paths, timeout)
    except Exception as exc:
        for item in batch:
            label_stats[item["true_label"]]["failed"] += 1
            save_failure_result(item["output_path"], item["image_path"], str(exc))
            print(
                f"[FAIL] {item['index']}/{image_count} {item['image_path']} error={exc}",
                file=sys.stderr,
            )
        return 0, 0, len(batch), 1

    if status_code == 413 and len(batch) > 1:
        mid = len(batch) // 2
        left = batch[:mid]
        right = batch[mid:]
        print(
            f"[SPLIT] {batch[0]['index']}-{batch[-1]['index']}/{image_count} "
            f"batch_size={len(batch)} status=413 -> {len(left)} + {len(right)}",
            file=sys.stderr,
        )
        left_counts = process_batch_with_retry(left, image_count, api_url, timeout, label_stats)
        right_counts = process_batch_with_retry(right, image_count, api_url, timeout, label_stats)
        return (
            left_counts[0] + right_counts[0],
            left_counts[1] + right_counts[1],
            left_counts[2] + right_counts[2],
            left_counts[3] + right_counts[3],
        )

    response_dict = response_json if isinstance(response_json, dict) else None
    results = response_dict.get("results") if isinstance(response_dict, dict) else None

    if (
        response_dict is None
        or status_code != 200
        or not response_dict.get("success", False)
        or not isinstance(results, list)
        or len(results) != len(batch)
    ):
        message = body_text
        if response_dict is not None and "error" in response_dict:
            message = str(response_dict["error"])

        if status_code == 413 and len(batch) == 1:
            message = f"413 Request Entity Too Large: {message}"

        for item in batch:
            label_stats[item["true_label"]]["failed"] += 1
            save_failure_result(item["output_path"], item["image_path"], message)
            print(
                f"[FAIL] {item['index']}/{image_count} {item['image_path']} "
                f"status={status_code} error={message}",
                file=sys.stderr,
            )
        return 0, 0, len(batch), 1

    batch_correct, batch_incorrect, batch_failed = handle_batch_results(
        batch,
        results,
        image_count,
        label_stats,
    )

    batch_elapsed = time.time() - batch_t0
    print(
        f"[BATCH] {batch[0]['index']}-{batch[-1]['index']}/{image_count} "
        f"batch_size={len(batch)} time={batch_elapsed:.2f}s"
    )
    return batch_correct, batch_incorrect, batch_failed, 1


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1

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

    tasks = []
    skipped = 0
    total = 0

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
        tasks.append({
            "index": index,
            "image_path": image_path,
            "true_label": true_label,
            "output_path": output_path,
        })

    correct = 0
    incorrect = 0
    failed = 0
    total_batches = 0

    for batch in chunked(tasks, args.batch_size):
        batch_correct, batch_incorrect, batch_failed, batch_count = process_batch_with_retry(
            batch,
            len(image_paths),
            args.api_url,
            args.timeout,
            label_stats,
        )
        correct += batch_correct
        incorrect += batch_incorrect
        failed += batch_failed
        total_batches += batch_count

    evaluated = correct + incorrect
    accuracy = (correct / evaluated * 100.0) if evaluated else 0.0

    print("\nSummary")
    print(f"API URL: {args.api_url}")
    print(f"Input folder: {input_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Batches sent: {total_batches}")
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
