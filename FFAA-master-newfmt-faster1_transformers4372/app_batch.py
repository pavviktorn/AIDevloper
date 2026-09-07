import base64
import binascii
import json
import os
import queue
import random
import re
import string
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import transformers
from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from transformers import AutoTokenizer, CLIPProcessor
from werkzeug.utils import secure_filename

transformers.logging.set_verbosity_error()

from llava.constants import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    IMAGE_PLACEHOLDER,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token
from llava.model import LlavaLlamaForCausalLM
from mids.selector import make_decision_batch
from utils.file_utils import decode_response, get_jsonfmt, mask_result, read_txt_file
from yolo11_cls_onnx import Yolo11ClsONNX


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_LIVENESS_PROMPT = "The image is a human face image. Is it real or fake? Why?"
DEFAULT_MODEL_PATH = "checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_0"
DEFAULT_MIDS_PATH = "checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_0/mids.pth"
IMAGE_FORMAT_TO_EXT = {
    "JPEG": ".jpg",
    "JPG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "BMP": ".bmp",
}

DEVICE_ID = int(os.environ.get("APP_BATCH_DEVICE", "0"))
MODEL_PATH = os.environ.get("APP_BATCH_MODEL_PATH", DEFAULT_MODEL_PATH)
MIDS_PATH = os.environ.get("APP_BATCH_MIDS_PATH", DEFAULT_MIDS_PATH)
T5_MODEL_PATH = os.environ.get("APP_BATCH_T5_MODEL_PATH", "models/t5-base")
CLIP_MODEL_PATH = os.environ.get("APP_BATCH_CLIP_MODEL_PATH", "models/clip-vit-large-patch14-336")
PROMPT_LIST_PATH = os.environ.get("APP_BATCH_PROMPT_LIST", "playground/prompts.txt")
DET_MODEL_PATH = os.environ.get(
    "APP_BATCH_ROT_DET_MODEL",
    "./rot_det_model/yolo11n-rotation2/weights/best.onnx",
)
BATCH_WINDOW_SEC = float(os.environ.get("APP_BATCH_WINDOW_SEC", "0.2"))
MAX_BATCH_SIZE = int(os.environ.get("APP_BATCH_MAX_SIZE", "64"))
REQUEST_TIMEOUT_SEC = float(os.environ.get("APP_BATCH_REQUEST_TIMEOUT_SEC", "180"))
SAVE_REQUEST_IMAGES = os.environ.get("APP_BATCH_SAVE_IMAGES", "0").lower() in {"1", "true", "yes"}
G_CROP = int(os.environ.get("APP_BATCH_CROP", "0"))
MISTRAL_GENERATIONS = int(os.environ.get("APP_BATCH_MISTRAL_GENERATIONS", "3"))
MAX_CONTENT_LENGTH_MB = int(os.environ.get("MAX_CONTENT_LENGTH_MB", "1024"))


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for app_batch.py")
if MISTRAL_GENERATIONS < 1 or MISTRAL_GENERATIONS > 3:
    raise RuntimeError("APP_BATCH_MISTRAL_GENERATIONS must be 1, 2, or 3")

torch.cuda.set_device(DEVICE_ID)
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)


def load_llava(model_path: str, device_id: int):
    kwargs = {
        "device_map": device_id,
        "torch_dtype": torch.float16,
        "use_flash_attention_2": True,
    }
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
        **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, assign=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vision_tower = model.get_vision_tower()
    image_processor = vision_tower.image_processor
    return model, image_processor, tokenizer


def load_mids(mids_path: str, device_id: int):
    from mids.mids_arch import MIDS

    model = MIDS(text_model_path=T5_MODEL_PATH, image_model_path=CLIP_MODEL_PATH)
    model_state_dict = model.state_dict()
    finetuned_state_dict = torch.load(mids_path, map_location="cpu")
    finetuned_state_dict = {
        key.replace("module.", ""): value for key, value in finetuned_state_dict.items()
    }
    model_state_dict.update(finetuned_state_dict)
    model.load_state_dict(model_state_dict)
    return model.to(dtype=torch.float32, device=torch.device(f"cuda:{device_id}"))


def get_llava_prompt(model, qs: str, conv_mode: str = "v1") -> str:
    image_token_se = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN
    if IMAGE_PLACEHOLDER in qs:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = re.sub(IMAGE_PLACEHOLDER, image_token_se, qs)
        else:
            qs = re.sub(IMAGE_PLACEHOLDER, DEFAULT_IMAGE_TOKEN, qs)
    else:
        if getattr(model.config, "mm_use_im_start_end", False):
            qs = image_token_se + "\n" + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def build_llava_batch_input_ids(model, tokenizer, qs_list: list[str], conv_mode: str):
    ids = []
    for qs in qs_list:
        prompt = get_llava_prompt(model, qs, conv_mode)
        tokens = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        )
        if tokens.ndim > 1:
            tokens = tokens.squeeze(0)
        ids.append(tokens)

    input_ids = torch.nn.utils.rnn.pad_sequence(
        ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    )
    return input_ids.to(model.device)


@torch.inference_mode()
def get_llava_answer_batch(
    model,
    tokenizer,
    image_processor,
    images: list[Any],
    prompts: list[str],
    temperature: float,
    top_p: Optional[float],
    num_beams: int,
    max_new_tokens: int,
    per_sample_generate_num: int,
    conv_mode: str = "v1",
) -> list[list[str]]:
    if len(images) != len(prompts):
        raise ValueError("images and prompts must have the same length")
    if not images:
        return []

    image_sizes = [image.size for image in images]
    image_tensor = process_images(
        images,
        image_processor,
        model.config,
    ).to(model.device, dtype=torch.float16)

    outputs = [[] for _ in images]
    condition_prompt = "This is a _ human face. What evidence do you have?"

    def run_generation(qs_list: list[str]) -> list[str]:
        input_ids = build_llava_batch_input_ids(model, tokenizer, qs_list, conv_mode)
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=top_p,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            attention_mask=(input_ids != tokenizer.pad_token_id).long(),
            pad_token_id=tokenizer.pad_token_id,
        )
        return [
            text.strip()
            for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        ]

    decoded = run_generation(prompts)
    for idx, text in enumerate(decoded):
        outputs[idx].append(text)

    if per_sample_generate_num <= 1:
        return outputs

    prompts_2 = []
    prompts_3 = []
    for answers in outputs:
        try:
            response_json, _ = decode_response(answers[0])
            answer_result = response_json["Analysis result"].lower()
        except Exception:
            answer_result = "fake"

        if answer_result == "real":
            prompts_2.append(condition_prompt.replace("_", "fake"))
            prompts_3.append(condition_prompt.replace("_", "real"))
        else:
            prompts_2.append(condition_prompt.replace("_", "real"))
            prompts_3.append(condition_prompt.replace("_", "fake"))

    decoded = run_generation(prompts_2)
    for idx, text in enumerate(decoded):
        outputs[idx].append(text)

    if per_sample_generate_num <= 2:
        return outputs

    decoded = run_generation(prompts_3)
    for idx, text in enumerate(decoded):
        outputs[idx].append(text)

    return [answers[:per_sample_generate_num] for answers in outputs]


print(f"[LOAD] LLaVA checkpoint: {MODEL_PATH}")
mistral_model, mistral_image_processor, mistral_tokenizer = load_llava(MODEL_PATH, DEVICE_ID)
mistral_model = mistral_model.to(torch.device(f"cuda:{DEVICE_ID}"))
mistral_model.eval()

print(f"[LOAD] MIDS checkpoint: {MIDS_PATH}")
t5_tokenizer = AutoTokenizer.from_pretrained(T5_MODEL_PATH, use_fast=False, legacy=False)
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_PATH)
mids = load_mids(MIDS_PATH, DEVICE_ID)
mids.eval()

print(f"[LOAD] Rotation detector: {DET_MODEL_PATH}")
det_model = Yolo11ClsONNX(
    onnx_path=DET_MODEL_PATH,
    imgsz=224,
    class_names=["0", "180", "270", "90"],
)


def get_random_string() -> str:
    return "".join(random.choice(string.ascii_letters) for _ in range(8))


def load_liveness_prompt() -> str:
    try:
        prompt_list = read_txt_file(PROMPT_LIST_PATH)
    except FileNotFoundError:
        prompt_list = []
    return prompt_list[0] if prompt_list else DEFAULT_LIVENESS_PROMPT


LIVENESS_PROMPT = load_liveness_prompt()


def get_liveness_prompt() -> str:
    return LIVENESS_PROMPT


def get_request_image_subdir(ipaddr: Optional[str]) -> Path:
    if ipaddr:
        ipaddr = ipaddr.split(",")[0].strip()
    if not ipaddr:
        ipaddr = "unknown"
    subdir = APP_ROOT / "images" / ipaddr
    subdir.mkdir(parents=True, exist_ok=True)
    return subdir


def save_request_image_bytes(image_bytes: bytes, suffix: str, ipaddr: Optional[str]) -> str:
    fname = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + suffix
    file_path = get_request_image_subdir(ipaddr) / fname
    file_path.write_bytes(image_bytes)
    return str(file_path)


def decode_image_bytes(image_bytes: bytes) -> tuple[Image.Image, np.ndarray, str]:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.load()
            image_format = (image.format or "").upper()
            pil_image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("Failed to decode image") from exc

    suffix = IMAGE_FORMAT_TO_EXT.get(image_format, ".png")
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    cv_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if cv_bgr is None:
        cv_rgb = np.asarray(pil_image)
        cv_bgr = cv2.cvtColor(cv_rgb, cv2.COLOR_RGB2BGR)
    return pil_image, cv_bgr, suffix


def base64_to_image_bytes(image_base64: str) -> bytes:
    if "," in image_base64:
        image_base64 = image_base64.split(",", 1)[1]
    try:
        return base64.b64decode(image_base64, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Invalid base64 image") from exc


def crop_face(image: Image.Image):
    return image


def get_rotated_angle(cv_bgr: np.ndarray) -> int:
    rot_angle, _conf = det_model.predict(cv_bgr)
    return rot_angle


def safe_parse_answer(answer: str) -> Any:
    try:
        return get_jsonfmt(answer)
    except Exception:
        parsed: dict[str, Any] = {}
        for line in answer.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if not key:
                continue
            try:
                parsed[key] = float(value) if re.fullmatch(r"\d+\.\d+", value) else value
            except Exception:
                parsed[key] = value
        return parsed if parsed else answer


def format_answer_json(answer_json: dict[str, Any]) -> str:
    data = dict(answer_json)
    data.pop("Probability", None)
    key_order = [
        "Image description",
        "Forgery reasoning",
        "Analysis result",
        "Forgery type",
        "Match score",
        "Difficulty",
    ]
    return "\n".join(f"{key}: {data[key]}" for key in key_order if key in data)


def build_face_liveness_string_response(answer: str) -> dict[str, Any]:
    return {
        "success": True,
        "face_liveness": json.dumps(safe_parse_answer(answer), indent=2, ensure_ascii=False),
    }


def build_face_liveness_object_response(answer: str) -> dict[str, Any]:
    return {
        "success": True,
        "face_liveness": safe_parse_answer(answer),
    }


def finalize_best_answer(
    answers: list[str],
    answers_result: list[str],
    best_answer_idx: int,
    match_score: float,
) -> tuple[str, int]:
    if len(set(answers_result)) == 1:
        qs_difficulty = "easy"
    else:
        qs_difficulty = "hard"

    best_answer_json, _ = decode_response(answers[best_answer_idx])
    if best_answer_json["Analysis result"].lower() == "real":
        best_answer_cls = 0
        probability = float(best_answer_json.get("Probability", 1.0))
        if probability < 0.8 or (match_score < 0.99 and match_score > 0.8):
            if qs_difficulty == "hard":
                selected_json = None
                for idx in range(len(answers_result) - 1, -1, -1):
                    if answers_result[idx] == "fake":
                        selected_json, _ = decode_response(answers[idx])
                        break
                if selected_json is not None:
                    best_answer_json["Forgery type"] = selected_json["Forgery type"]
                    best_answer_json["Forgery reasoning"] = (
                        selected_json["Forgery reasoning"]
                        + ' It\'s close to "real", but not completely certain.'
                    )
                else:
                    best_answer_json["Forgery reasoning"] = (
                        best_answer_json["Forgery reasoning"]
                        + ' It\'s close to "real", but not completely certain.'
                    )
                best_answer_json["Analysis result"] = "ambiguous"
            else:
                best_answer_json["Analysis result"] = "ambiguous"
                best_answer_json["Forgery reasoning"] = (
                    best_answer_json["Forgery reasoning"]
                    + ' It\'s close to "real", but not completely certain.'
                )
        elif match_score <= 0.8:
            best_answer_json["Forgery type"] = "ambiguous"
            best_answer_json["Analysis result"] = "likely_fake"
            if qs_difficulty == "hard":
                selected_json = None
                for idx in range(len(answers_result) - 1, -1, -1):
                    if answers_result[idx] == "fake":
                        selected_json, _ = decode_response(answers[idx])
                        break
                if selected_json is not None:
                    best_answer_json["Forgery type"] = selected_json["Forgery type"]
                    best_answer_json["Forgery reasoning"] = (
                        selected_json["Forgery reasoning"]
                        + ' It\'s closer to "fake", but not completely certain.'
                    )
                else:
                    best_answer_json["Forgery reasoning"] = (
                        best_answer_json["Forgery reasoning"]
                        + ' It\'s closer to "fake", but not completely certain.'
                    )
            else:
                best_answer_json["Forgery reasoning"] = (
                    best_answer_json["Forgery reasoning"]
                    + ' It\'s closer to "fake", but not completely certain.'
                )
    else:
        best_answer_cls = 3

    best_answer_json["Match score"] = f"{match_score:.4f}"
    best_answer_json["Difficulty"] = qs_difficulty
    return format_answer_json(best_answer_json), best_answer_cls


def mids_shape_for_generations(generation_count: int) -> tuple[int, int]:
    if generation_count >= 3:
        return 1, 1
    if generation_count == 2:
        return 0, 1
    return 0, 0


@dataclass
class BatchJob:
    pil_image: Image.Image
    cv_bgr: np.ndarray
    save_path: Optional[str] = None
    created_at: float = field(default_factory=time.monotonic)
    done: threading.Event = field(default_factory=threading.Event)
    result: Optional[dict[str, Any]] = None


class RequestBatcher:
    def __init__(
        self,
        batch_window_sec: float,
        max_batch_size: int,
        request_timeout_sec: float,
    ) -> None:
        self.batch_window_sec = max(0.0, batch_window_sec)
        self.max_batch_size = max(1, max_batch_size)
        self.request_timeout_sec = request_timeout_sec
        self.pending: deque[BatchJob] = deque()
        self.condition = threading.Condition()
        self.stopped = False
        self.stats_lock = threading.Lock()
        self.stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "batches": 0,
            "max_batch_size_seen": 0,
            "last_batch_size": 0,
            "last_queue_depth": 0,
            "last_batch_wait_ms": 0.0,
            "last_batch_infer_ms": 0.0,
        }
        self.worker = threading.Thread(target=self._worker_loop, name="request-batcher", daemon=True)
        self.worker.start()

    def submit(self, job: BatchJob) -> BatchJob:
        with self.condition:
            if self.stopped:
                job.result = {"success": False, "error": "Batch processor is stopped"}
                job.done.set()
                return job
            self.pending.append(job)
            with self.stats_lock:
                self.stats["submitted"] += 1
                self.stats["last_queue_depth"] = len(self.pending)
            self.condition.notify()
        return job

    def submit_many(self, jobs: list[BatchJob]) -> list[BatchJob]:
        if not jobs:
            return jobs
        with self.condition:
            if self.stopped:
                for job in jobs:
                    job.result = {"success": False, "error": "Batch processor is stopped"}
                    job.done.set()
                return jobs
            self.pending.extend(jobs)
            with self.stats_lock:
                self.stats["submitted"] += len(jobs)
                self.stats["last_queue_depth"] = len(self.pending)
            self.condition.notify()
        return jobs

    def wait(self, job: BatchJob) -> dict[str, Any]:
        if not job.done.wait(self.request_timeout_sec):
            return {"success": False, "error": "Batch inference timeout"}
        return job.result or {"success": False, "error": "Unknown batch inference error"}

    def snapshot_stats(self) -> dict[str, Any]:
        with self.condition:
            pending_count = len(self.pending)
        with self.stats_lock:
            stats = dict(self.stats)
        stats.update(
            {
                "pending": pending_count,
                "batch_window_sec": self.batch_window_sec,
                "max_batch_size": self.max_batch_size,
                "request_timeout_sec": self.request_timeout_sec,
            }
        )
        return stats

    def _take_batch(self) -> list[BatchJob]:
        with self.condition:
            while not self.pending and not self.stopped:
                self.condition.wait()
            if self.stopped:
                return []

            first_job_time = self.pending[0].created_at
            deadline = first_job_time + self.batch_window_sec
            while len(self.pending) < self.max_batch_size and not self.stopped:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self.condition.wait(timeout=remaining)

            batch_size = min(len(self.pending), self.max_batch_size)
            batch = [self.pending.popleft() for _ in range(batch_size)]
            with self.stats_lock:
                self.stats["last_queue_depth"] = len(self.pending)
            return batch

    def _worker_loop(self) -> None:
        while True:
            batch = self._take_batch()
            if not batch:
                return

            wait_ms = (time.monotonic() - min(job.created_at for job in batch)) * 1000.0
            infer_start = time.monotonic()
            results = self._infer_batch_with_retry(batch)
            infer_ms = (time.monotonic() - infer_start) * 1000.0

            for job, result in zip(batch, results):
                job.result = result
                job.done.set()

            failed = sum(1 for result in results if not result.get("success", False))
            with self.stats_lock:
                self.stats["completed"] += len(results)
                self.stats["failed"] += failed
                self.stats["batches"] += 1
                self.stats["last_batch_size"] = len(batch)
                self.stats["max_batch_size_seen"] = max(
                    self.stats["max_batch_size_seen"],
                    len(batch),
                )
                self.stats["last_batch_wait_ms"] = wait_ms
                self.stats["last_batch_infer_ms"] = infer_ms

            print(
                f"[BATCH] size={len(batch)} wait_ms={wait_ms:.1f} "
                f"infer_ms={infer_ms:.1f} failed={failed}",
                flush=True,
            )

    def _infer_batch_with_retry(self, jobs: list[BatchJob]) -> list[dict[str, Any]]:
        try:
            return run_liveness_batch(jobs)
        except torch.cuda.OutOfMemoryError as exc:
            torch.cuda.empty_cache()
            if len(jobs) > 1:
                mid = len(jobs) // 2
                return self._infer_batch_with_retry(jobs[:mid]) + self._infer_batch_with_retry(jobs[mid:])
            return [{"success": False, "error": f"CUDA out of memory: {exc}"}]
        except Exception as exc:
            if len(jobs) > 1:
                mid = len(jobs) // 2
                return self._infer_batch_with_retry(jobs[:mid]) + self._infer_batch_with_retry(jobs[mid:])
            return [{"success": False, "error": str(exc)}]


def run_liveness_batch(jobs: list[BatchJob]) -> list[dict[str, Any]]:
    prompt = get_liveness_prompt()
    device = torch.device(f"cuda:{DEVICE_ID}")

    results: list[Optional[dict[str, Any]]] = [None] * len(jobs)
    valid_indices: list[int] = []
    valid_images: list[Image.Image] = []

    for idx, job in enumerate(jobs):
        angle = get_rotated_angle(job.cv_bgr)
        if angle > 0:
            results[idx] = {
                "success": False,
                "error": "The image appears to be rotated. Please try again with a straightened image.",
            }
            continue

        image = job.pil_image
        if G_CROP == 1:
            image = crop_face(image)
            if image is None:
                results[idx] = {"success": False, "error": "No face detected"}
                continue

        valid_indices.append(idx)
        valid_images.append(image)

    if not valid_images:
        return [result or {"success": False, "error": "Unknown input error"} for result in results]

    with torch.inference_mode():
        answers_per_image = get_llava_answer_batch(
            mistral_model,
            mistral_tokenizer,
            mistral_image_processor,
            valid_images,
            [prompt] * len(valid_images),
            temperature=0.0,
            top_p=None,
            num_beams=1,
            max_new_tokens=512,
            per_sample_generate_num=MISTRAL_GENERATIONS,
            conv_mode="v1",
        )

    flattened_processed_answers: list[str] = []
    flattened_answers_result: list[str] = []
    mids_indices: list[int] = []
    mids_images: list[Image.Image] = []
    mids_answers_per_image: list[list[str]] = []
    answers_result_per_image: list[list[str]] = []

    expected_count = max(1, MISTRAL_GENERATIONS)
    for batch_idx, answers in enumerate(answers_per_image):
        original_idx = valid_indices[batch_idx]
        if len(answers) != expected_count:
            results[original_idx] = {
                "success": False,
                "error": f"expected {expected_count} generated answers, got {len(answers)}",
            }
            continue

        answers_result: list[str] = []
        processed_answers: list[str] = []
        try:
            for answer in answers:
                masked_answer, answer_res = mask_result(answer)
                answers_result.append(answer_res)
                processed_answers.append(masked_answer)
        except Exception as exc:
            results[original_idx] = {"success": False, "error": str(exc)}
            continue

        mids_indices.append(original_idx)
        mids_images.append(valid_images[batch_idx])
        mids_answers_per_image.append(answers)
        answers_result_per_image.append(answers_result)
        flattened_answers_result.extend(answers_result)
        flattened_processed_answers.extend(processed_answers)

    if not mids_images:
        return [result or {"success": False, "error": "Unknown inference error"} for result in results]

    n_condition, m_condition = mids_shape_for_generations(expected_count)
    with torch.inference_mode():
        input_images = clip_processor(images=mids_images, return_tensors="pt")["pixel_values"]
        answer_ids = t5_tokenizer(
            flattened_processed_answers,
            return_tensors="pt",
            padding="longest",
            max_length=t5_tokenizer.model_max_length,
            truncation=True,
        )
        logits = mids(
            answer_ids.to(device),
            input_images.to(device),
            None,
            len(mids_images),
            n_condition,
            m_condition,
        )["logits"]
        scores = F.softmax(logits, dim=2)
        best_answer_idxs, _preds, match_scores, _forgery_scores = make_decision_batch(
            flattened_answers_result,
            scores,
            chunk_size=expected_count,
        )

    for batch_idx, original_idx in enumerate(mids_indices):
        try:
            best_answer, _cls = finalize_best_answer(
                mids_answers_per_image[batch_idx],
                answers_result_per_image[batch_idx],
                int(best_answer_idxs[batch_idx]),
                float(match_scores[batch_idx]),
            )
        except Exception as exc:
            results[original_idx] = {"success": False, "error": str(exc)}
            continue

        save_path = jobs[original_idx].save_path
        if save_path:
            text_path = str(Path(save_path).with_suffix(".txt"))
            with open(text_path, "w", encoding="utf-8") as handle:
                handle.write(best_answer)

        results[original_idx] = {"success": True, "face_liveness": best_answer}

    return [result or {"success": False, "error": "Unknown inference error"} for result in results]


batcher = RequestBatcher(
    batch_window_sec=BATCH_WINDOW_SEC,
    max_batch_size=MAX_BATCH_SIZE,
    request_timeout_sec=REQUEST_TIMEOUT_SEC,
)

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH_MB * 1024 * 1024


@app.after_request
def set_response_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route("/")
def homepage():
    try:
        return render_template("home_two.html")
    except Exception:
        return "app_batch is running"


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"success": True, "status": "ok", "batcher": batcher.snapshot_stats()})


@app.route("/batch_stats", methods=["GET"])
def batch_stats():
    return jsonify({"success": True, "batcher": batcher.snapshot_stats()})


def make_job_from_bytes(image_bytes: bytes, ipaddr: Optional[str]) -> BatchJob:
    pil_image, cv_bgr, suffix = decode_image_bytes(image_bytes)
    save_path = None
    if SAVE_REQUEST_IMAGES:
        save_path = save_request_image_bytes(image_bytes, suffix, ipaddr)
    return BatchJob(pil_image=pil_image, cv_bgr=cv_bgr, save_path=save_path)


def wait_or_504(job: BatchJob):
    result = batcher.wait(job)
    if result.get("success", False):
        return result, 200
    if result.get("error") == "Batch inference timeout":
        return result, 504
    return result, 200


def wait_until(job: BatchJob, deadline: float) -> dict[str, Any]:
    timeout = max(0.0, deadline - time.monotonic())
    if not job.done.wait(timeout):
        return {"success": False, "error": "Batch inference timeout"}
    return job.result or {"success": False, "error": "Unknown batch inference error"}


@app.route("/face_liveness", methods=["POST"])
def receive_face():
    if "face" not in request.files:
        return jsonify({"error": "no face image file."}), 400

    face_image = request.files["face"]
    if face_image.filename == "":
        return jsonify({"error": "no face image file."}), 400

    _file_name = secure_filename(face_image.filename)
    image_bytes = face_image.read()
    try:
        job = make_job_from_bytes(image_bytes, request.headers.get("X-Forwarded-For", request.remote_addr))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    batcher.submit(job)
    result, status_code = wait_or_504(job)
    if not result.get("success", False):
        return jsonify(result), status_code

    return jsonify(build_face_liveness_string_response(result["face_liveness"]))


@app.route("/face_liveness_base64", methods=["POST"])
def receive_face_base64():
    data = request.get_json(silent=True)
    if not data or "image_base64" not in data:
        return jsonify({"error": "image_base64 is required"}), 400

    image_base64 = data["image_base64"]
    if not isinstance(image_base64, str) or not image_base64.strip():
        return jsonify({"error": "image_base64 must be a non-empty string"}), 400

    try:
        image_bytes = base64_to_image_bytes(image_base64)
        job = make_job_from_bytes(image_bytes, request.headers.get("X-Forwarded-For", request.remote_addr))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    batcher.submit(job)
    result, status_code = wait_or_504(job)
    if not result.get("success", False):
        return jsonify(result), status_code

    return jsonify(build_face_liveness_object_response(result["face_liveness"]))


@app.route("/face_liveness_base64_batch", methods=["POST"])
def receive_face_base64_batch():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "images_base64 is required"}), 400

    images_base64 = data.get("images_base64")
    if images_base64 is None:
        images_base64 = data.get("image_base64_list")

    if not isinstance(images_base64, list) or len(images_base64) == 0:
        return jsonify({"error": "images_base64 must be a non-empty list"}), 400

    ipaddr = request.headers.get("X-Forwarded-For", request.remote_addr)
    results: list[Optional[dict[str, Any]]] = [None] * len(images_base64)
    jobs: list[tuple[int, BatchJob]] = []

    for idx, image_base64 in enumerate(images_base64):
        if not isinstance(image_base64, str) or not image_base64.strip():
            results[idx] = {"success": False, "error": "image_base64 must be a non-empty string"}
            continue
        try:
            image_bytes = base64_to_image_bytes(image_base64)
            job = make_job_from_bytes(image_bytes, ipaddr)
        except ValueError as exc:
            results[idx] = {"success": False, "error": str(exc)}
            continue
        jobs.append((idx, job))

    batcher.submit_many([job for _idx, job in jobs])
    deadline = time.monotonic() + REQUEST_TIMEOUT_SEC
    for idx, job in jobs:
        result = wait_until(job, deadline)
        if result.get("success", False):
            results[idx] = build_face_liveness_object_response(result["face_liveness"])
        else:
            results[idx] = result

    for idx, result in enumerate(results):
        if result is None:
            results[idx] = {"success": False, "error": "Unknown input error"}

    return jsonify({"success": True, "results": results})


def get_ssl_context():
    cert_path = os.environ.get("SSL_CERT_PATH", "certs/cert.pem")
    key_path = os.environ.get("SSL_KEY_PATH", "certs/key.pem")

    if (
        os.path.isfile(cert_path)
        and os.path.isfile(key_path)
        and os.access(cert_path, os.R_OK)
        and os.access(key_path, os.R_OK)
    ):
        return (cert_path, key_path)

    print(f"SSL cert/key not readable ({cert_path}, {key_path}); using ad-hoc self-signed cert.")
    return "adhoc"


if __name__ == "__main__":
    app.run(
        host=os.environ.get("APP_BATCH_HOST", "0.0.0.0"),
        port=int(os.environ.get("APP_BATCH_PORT", "3000")),
        ssl_context=get_ssl_context(),
        threaded=True,
        use_reloader=False,
    )
