#!/usr/bin/env python3
import argparse
import base64
import json
import ssl
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import error, request
from urllib.parse import urlparse

from test_folder import (
    IMAGE_EXTENSIONS,
    LABELS,
    build_label_stats,
    dump_text_file,
    extract_analysis_result,
    extract_match_score,
    normalize_prediction_label,
    parse_json_text,
    response_to_text,
)


# DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/AxonLabs/fake/level2/3. Textile Cloth attacks (2 full sets)"
DEFAULT_INPUT_DIR = "/datasets/work/vLLM/data/Axonlabs_replay_samples"
DEFAULT_BATCH_API_URL = "https://127.0.0.1:3000/face_liveness_base64_batch"
# DEFAULT_MISS_DIR = "/datasets/work/vLLM/data/miss_unidata"
DEFAULT_MISS_DIR = "/datasets/work/vLLM/data/miss_ibeta2_replay"
LOCAL_HTTPS_HOSTS = {"127.0.0.1", "localhost", "::1"}
VIDEO_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
}
JPEG_QUALITY = 95
cv2: Any = None
np: Any = None


def load_image_dependencies() -> bool:
    global cv2
    global np

    if cv2 is not None and np is not None:
        return True

    try:
        import cv2 as cv2_module
        import numpy as np_module
    except ModuleNotFoundError as exc:
        print(
            f"Missing required Python package: {exc.name}. "
            "Activate the environment with opencv-python and numpy installed.",
            file=sys.stderr,
        )
        return False

    cv2 = cv2_module
    np = np_module
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process still images and every frame of every video under a folder "
            "with the batched face_liveness_base64_batch endpoint. Successful "
            "misses are copied to miss_unidata with mirrored source structure."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help=f"Folder to scan for image/video files. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--api-url",
        default=DEFAULT_BATCH_API_URL,
        help=f"Batch endpoint URL. Default: {DEFAULT_BATCH_API_URL}",
    )
    parser.add_argument(
        "--miss-dir",
        default=DEFAULT_MISS_DIR,
        help=f"Folder used to save missed images. Default: {DEFAULT_MISS_DIR}",
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
        default=160,
        help="Number of images/frames per batch request. Default: 24",
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
        help=(
            "Skip still images with an existing .txt output and skip videos with "
            "an existing .frames.jsonl output."
        ),
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for any HTTPS endpoint.",
    )
    parser.add_argument(
        "--verify-tls",
        action="store_true",
        help=(
            "Verify TLS certificates even for localhost HTTPS endpoints. By "
            "default, localhost verification is disabled so Flask's self-signed "
            "development certificate works."
        ),
    )
    return parser.parse_args()


def build_ssl_context(
    api_url: str,
    insecure: bool,
    verify_tls: bool,
) -> Optional[ssl.SSLContext]:
    parsed = urlparse(api_url)
    if parsed.scheme.lower() != "https":
        return None

    should_skip_verify = insecure or (
        not verify_tls and parsed.hostname in LOCAL_HTTPS_HOSTS
    )
    if not should_skip_verify:
        return None

    return ssl._create_unverified_context()


def iter_media_files(root_dir: Path) -> Iterable[Path]:
    for path in sorted(root_dir.rglob("*")):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_EXTENSIONS or suffix in VIDEO_EXTENSIONS:
            yield path


def build_batch_request_body(batch: list[dict[str, Any]]) -> bytes:
    images_base64 = []
    for item in batch:
        images_base64.append(base64.b64encode(item["payload_bytes"]).decode("ascii"))
    payload = {"images_base64": images_base64}
    return json.dumps(payload).encode("utf-8")


def post_batch(
    api_url: str,
    batch: list[dict[str, Any]],
    timeout: float,
    ssl_context: Optional[ssl.SSLContext],
) -> tuple[int, str, Any]:
    payload = build_batch_request_body(batch)
    req = request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=timeout, context=ssl_context) as response:
            status_code = response.getcode()
            body_text = response.read().decode("utf-8", errors="replace")
    except error.HTTPError as exc:
        status_code = exc.code
        body_text = exc.read().decode("utf-8", errors="replace")
    except error.URLError as exc:
        raise RuntimeError(f"request failed: {exc.reason}") from exc

    response_json = parse_json_text(body_text)
    return status_code, body_text, response_json


def get_label_index(path: Path) -> Optional[int]:
    for index, part in enumerate(path.parts[:-1]):
        lower_part = part.lower()
        if lower_part == "real" or lower_part == "fake":
            return index
    return None


def get_true_label_from_path(path: Path) -> Optional[str]:
    label_index = get_label_index(path)
    if label_index is None:
        return None
    return path.parts[label_index].lower()


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def get_source_output_path(source_path: Path) -> Path:
    suffix = source_path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return source_path.with_suffix(".txt")
    return Path(f"{source_path}.frames.jsonl")


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(item, ensure_ascii=False))
        handle.write("\n")


def relative_parent_for_source(source_path: Path, input_dir: Path) -> Path:
    try:
        return source_path.relative_to(input_dir).parent
    except ValueError:
        label_index = get_label_index(source_path)
        if label_index is not None:
            return Path(*source_path.parts[label_index:-1])
        return source_path.parent


def infer_source_name(source_path: Path, input_dir: Path) -> str:
    try:
        parent_parts = source_path.relative_to(input_dir).parent.parts
    except ValueError:
        parent_parts = source_path.parent.parts

    for part in reversed(parent_parts):
        if part.lower() not in LABELS:
            return part
    return source_path.stem


def unique_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    counter = 1
    while True:
        next_candidate = candidate.with_name(f"{stem}_{counter}{suffix}")
        if not next_candidate.exists():
            return next_candidate
        counter += 1


def encode_image_bytes_as_jpeg(image_bytes: bytes) -> bytes:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("failed to decode image")

    success, jpeg = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
    )
    if not success:
        raise RuntimeError("failed to encode image as JPEG")
    return jpeg.tobytes()


def build_miss_output_path(item: dict[str, Any], miss_dir: Path) -> Path:
    candidate = miss_dir / item["relative_parent"] / item["miss_filename"]
    return unique_path(candidate)


def save_missed_image(item: dict[str, Any], miss_dir: Path) -> Path:
    output_path = build_miss_output_path(item, miss_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if item["kind"] == "video_frame":
        miss_bytes = item["payload_bytes"]
    else:
        miss_bytes = encode_image_bytes_as_jpeg(item["payload_bytes"])

    output_path.write_bytes(miss_bytes)
    return output_path


def build_video_frame_record(
    item: dict[str, Any],
    result: Any,
    analysis_result: Optional[str],
    predicted_label: str,
    match_score: Optional[str],
    miss_output_path: Optional[Path],
) -> dict[str, Any]:
    return {
        "source_video": str(item["source_path"]),
        "frame_index": item["frame_index"],
        "frame_name": item["miss_filename"],
        "true_label": item["true_label"],
        "analysis_result": analysis_result,
        "predicted_label": predicted_label,
        "match_score": match_score,
        "miss_copy": str(miss_output_path) if miss_output_path is not None else None,
        "response": result,
    }


def save_result_for_item(
    item: dict[str, Any],
    result: Any,
    analysis_result: Optional[str],
    predicted_label: str,
    match_score: Optional[str],
    miss_output_path: Optional[Path],
) -> None:
    if item["kind"] == "image":
        if isinstance(result, dict):
            dump_text_file(item["output_path"], response_to_text(result))
        else:
            dump_text_file(item["output_path"], json.dumps(result, ensure_ascii=False, indent=2))
        return

    append_jsonl(
        item["output_path"],
        build_video_frame_record(
            item,
            result,
            analysis_result,
            predicted_label,
            match_score,
            miss_output_path,
        ),
    )


def save_failure_result_for_item(item: dict[str, Any], message: str) -> None:
    error_json = {
        "success": False,
        "error": message,
        "source": str(item["source_path"]),
    }

    if item["kind"] == "image":
        dump_text_file(item["output_path"], response_to_text(error_json))
        return

    record = build_video_frame_record(
        item,
        error_json,
        None,
        "fake",
        None,
        None,
    )
    append_jsonl(item["output_path"], record)


def handle_batch_results(
    batch: list[dict[str, Any]],
    results: list[Any],
    miss_dir: Path,
    label_stats: dict[str, dict[str, int]],
) -> tuple[int, int, int, int]:
    correct = 0
    incorrect = 0
    failed = 0
    miss_saved = 0

    for item, result in zip(batch, results):
        analysis_result = extract_analysis_result(result)
        match_score = extract_match_score(result)
        predicted_label = normalize_prediction_label(analysis_result)

        if not isinstance(result, dict) or not result.get("success", False):
            save_result_for_item(item, result, analysis_result, predicted_label, match_score, None)
            failed += 1
            label_stats[item["true_label"]]["failed"] += 1
            error_message = "N/A"
            if isinstance(result, dict):
                error_message = str(result.get("error", "N/A"))
            shown_result = analysis_result if analysis_result is not None else "N/A"
            shown_match_score = match_score if match_score is not None else "N/A"
            print(
                f"[FAIL] item={item['index']} {item['display_path']} "
                f"error={error_message} analysis={shown_result} "
                f"match_score={shown_match_score} predicted={predicted_label} "
                f"true={item['true_label']}",
                file=sys.stderr,
            )
            continue

        label_stats[item["true_label"]]["evaluated"] += 1
        miss_output_path = None
        if predicted_label == item["true_label"]:
            correct += 1
            label_stats[item["true_label"]]["correct"] += 1
            state = "OK"
        else:
            incorrect += 1
            label_stats[item["true_label"]]["miss"] += 1
            state = "MISS"
            try:
                miss_output_path = save_missed_image(item, miss_dir)
                miss_saved += 1
            except Exception as exc:
                print(
                    f"[MISS-SAVE-FAIL] item={item['index']} {item['display_path']} error={exc}",
                    file=sys.stderr,
                )

        save_result_for_item(
            item,
            result,
            analysis_result,
            predicted_label,
            match_score,
            miss_output_path,
        )

        shown_result = analysis_result if analysis_result is not None else "N/A"
        shown_match_score = match_score if match_score is not None else "N/A"
        extra = f" miss_copy={miss_output_path}" if miss_output_path is not None else ""
        print(
            f"[{state}] item={item['index']} {item['display_path']} "
            f"true={item['true_label']} analysis={shown_result} "
            f"match_score={shown_match_score} predicted={predicted_label}{extra}"
        )

    return correct, incorrect, failed, miss_saved


def process_batch_with_retry(
    batch: list[dict[str, Any]],
    api_url: str,
    timeout: float,
    miss_dir: Path,
    label_stats: dict[str, dict[str, int]],
    ssl_context: Optional[ssl.SSLContext],
) -> tuple[int, int, int, int, int]:
    batch_t0 = time.time()

    try:
        status_code, body_text, response_json = post_batch(api_url, batch, timeout, ssl_context)
    except Exception as exc:
        for item in batch:
            label_stats[item["true_label"]]["failed"] += 1
            save_failure_result_for_item(item, str(exc))
            print(
                f"[FAIL] item={item['index']} {item['display_path']} error={exc}",
                file=sys.stderr,
            )
        return 0, 0, len(batch), 0, 1

    if status_code == 413 and len(batch) > 1:
        mid = len(batch) // 2
        left = batch[:mid]
        right = batch[mid:]
        print(
            f"[SPLIT] items={batch[0]['index']}-{batch[-1]['index']} "
            f"batch_size={len(batch)} status=413 -> {len(left)} + {len(right)}",
            file=sys.stderr,
        )
        left_counts = process_batch_with_retry(
            left,
            api_url,
            timeout,
            miss_dir,
            label_stats,
            ssl_context,
        )
        right_counts = process_batch_with_retry(
            right,
            api_url,
            timeout,
            miss_dir,
            label_stats,
            ssl_context,
        )
        return (
            left_counts[0] + right_counts[0],
            left_counts[1] + right_counts[1],
            left_counts[2] + right_counts[2],
            left_counts[3] + right_counts[3],
            left_counts[4] + right_counts[4],
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
            save_failure_result_for_item(item, message)
            print(
                f"[FAIL] item={item['index']} {item['display_path']} "
                f"status={status_code} error={message}",
                file=sys.stderr,
            )
        return 0, 0, len(batch), 0, 1

    batch_correct, batch_incorrect, batch_failed, batch_miss_saved = handle_batch_results(
        batch,
        results,
        miss_dir,
        label_stats,
    )

    batch_elapsed = time.time() - batch_t0
    print(
        f"[BATCH] items={batch[0]['index']}-{batch[-1]['index']} "
        f"batch_size={len(batch)} time={batch_elapsed:.2f}s"
    )
    return batch_correct, batch_incorrect, batch_failed, batch_miss_saved, 1


def main() -> int:
    args = parse_args()

    if args.batch_size < 1:
        print("--batch-size must be at least 1", file=sys.stderr)
        return 1
    if args.insecure and args.verify_tls:
        print("--insecure and --verify-tls cannot be used together", file=sys.stderr)
        return 1

    ssl_context = build_ssl_context(args.api_url, args.insecure, args.verify_tls)
    if ssl_context is not None:
        print(
            f"[TLS] Certificate verification disabled for {args.api_url}",
            file=sys.stderr,
        )

    input_dir = Path(args.input_dir).expanduser().resolve()
    if not input_dir.exists():
        print(f"Input folder does not exist: {input_dir}", file=sys.stderr)
        return 1
    if not input_dir.is_dir():
        print(f"Input path is not a folder: {input_dir}", file=sys.stderr)
        return 1

    miss_dir = Path(args.miss_dir).expanduser().resolve()

    media_paths = []
    for path in iter_media_files(input_dir):
        if is_relative_to(path, miss_dir):
            continue
        media_paths.append(path)

    if not media_paths:
        print(f"No image/video files found under {input_dir}")
        return 0

    if not load_image_dependencies():
        return 1

    source_image_count = sum(1 for path in media_paths if path.suffix.lower() in IMAGE_EXTENSIONS)
    source_video_count = sum(1 for path in media_paths if path.suffix.lower() in VIDEO_EXTENSIONS)

    label_stats = build_label_stats()

    batch: list[dict[str, Any]] = []
    next_index = 0
    total_items = 0
    still_images_processed = 0
    video_frames_extracted = 0
    correct = 0
    incorrect = 0
    failed = 0
    miss_saved = 0
    total_batches = 0
    skipped_sources = 0
    failed_sources = 0

    def flush_batch_if_needed(force: bool = False) -> None:
        nonlocal batch
        nonlocal correct
        nonlocal incorrect
        nonlocal failed
        nonlocal miss_saved
        nonlocal total_batches

        if not batch:
            return
        if not force and len(batch) < args.batch_size:
            return

        (
            batch_correct,
            batch_incorrect,
            batch_failed,
            batch_miss_saved,
            batch_count,
        ) = process_batch_with_retry(
            batch,
            args.api_url,
            args.timeout,
            miss_dir,
            label_stats,
            ssl_context,
        )
        correct += batch_correct
        incorrect += batch_incorrect
        failed += batch_failed
        miss_saved += batch_miss_saved
        total_batches += batch_count
        batch = []

    for source_path in media_paths:
        true_label = get_true_label_from_path(source_path)
        if true_label is None:
            skipped_sources += 1
            print(f"[SKIP] {source_path} (no /real/ or /fake/ in path)")
            continue

        source_suffix = source_path.suffix.lower()
        output_path = get_source_output_path(source_path)
        if args.skip_existing and output_path.exists():
            skipped_sources += 1
            print(f"[SKIP] {source_path} ({output_path.name} already exists)")
            continue

        if source_suffix in IMAGE_EXTENSIONS:
            next_index += 1
            total_items += 1
            still_images_processed += 1
            label_stats[true_label]["found"] += 1
            label_stats[true_label]["processed"] += 1

            try:
                payload_bytes = source_path.read_bytes()
            except Exception as exc:
                item = {
                    "index": next_index,
                    "kind": "image",
                    "display_path": str(source_path),
                    "source_path": source_path,
                    "true_label": true_label,
                    "output_path": output_path,
                }
                failed += 1
                label_stats[true_label]["failed"] += 1
                save_failure_result_for_item(item, str(exc))
                print(f"[FAIL] item={next_index} {source_path} error={exc}", file=sys.stderr)
                continue

            item = {
                "index": next_index,
                "kind": "image",
                "display_path": str(source_path),
                "source_path": source_path,
                "true_label": true_label,
                "output_path": output_path,
                "payload_bytes": payload_bytes,
                "relative_parent": relative_parent_for_source(source_path, input_dir),
                "miss_filename": f"{infer_source_name(source_path, input_dir)}_{source_path.stem}.jpg",
            }
            batch.append(item)
            flush_batch_if_needed()
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("", encoding="utf-8")

        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            failed_sources += 1
            append_jsonl(
                output_path,
                {
                    "source_video": str(source_path),
                    "error": "failed to open video",
                    "success": False,
                },
            )
            print(f"[FAIL-VIDEO] {source_path} error=failed to open video", file=sys.stderr)
            continue

        frame_index = 0
        relative_parent = relative_parent_for_source(source_path, input_dir)
        video_name = source_path.stem

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            video_frames_extracted += 1
            next_index += 1
            total_items += 1
            label_stats[true_label]["found"] += 1
            label_stats[true_label]["processed"] += 1

            item = {
                "index": next_index,
                "kind": "video_frame",
                "display_path": f"{source_path}#frame={frame_index:06d}",
                "source_path": source_path,
                "true_label": true_label,
                "output_path": output_path,
                "frame_index": frame_index,
                "relative_parent": relative_parent,
                "miss_filename": f"{video_name}_{frame_index:06d}.jpg",
            }

            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )
            if not success:
                failed += 1
                label_stats[true_label]["failed"] += 1
                save_failure_result_for_item(item, "failed to encode frame as JPEG")
                print(
                    f"[FAIL-FRAME] item={next_index} {item['display_path']} "
                    f"error=failed to encode frame as JPEG",
                    file=sys.stderr,
                )
                continue

            item["payload_bytes"] = encoded.tobytes()
            batch.append(item)
            flush_batch_if_needed()

        cap.release()

        if frame_index == 0:
            failed_sources += 1
            append_jsonl(
                output_path,
                {
                    "source_video": str(source_path),
                    "error": "no frames extracted",
                    "success": False,
                },
            )
            print(f"[FAIL-VIDEO] {source_path} error=no frames extracted", file=sys.stderr)

    flush_batch_if_needed(force=True)

    evaluated = correct + incorrect
    accuracy = (correct / evaluated * 100.0) if evaluated else 0.0

    print("\nSummary")
    print(f"API URL: {args.api_url}")
    print(f"Input folder: {input_dir}")
    print(f"Miss folder: {miss_dir}")
    print(f"Batch size: {args.batch_size}")
    print(f"Source images found: {source_image_count}")
    print(f"Source videos found: {source_video_count}")
    print(f"Still images processed: {still_images_processed}")
    print(f"Video frames extracted: {video_frames_extracted}")
    print(f"Processed items: {total_items}")
    print(f"Evaluated: {evaluated}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Miss images saved: {miss_saved}")
    print(f"Failed items: {failed}")
    print(f"Failed video sources: {failed_sources}")
    print(f"Skipped sources: {skipped_sources}")
    print(f"Batches sent: {total_batches}")
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

    return 0 if failed == 0 and failed_sources == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
