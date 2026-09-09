import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import queue
import re
import threading
import time
import tkinter as tk
from tkinter import filedialog, ttk
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}

_CHECK_LIVE_FN: Optional[Callable[[str], Dict[str, object]]] = None
_JSON_FMT_FN: Optional[Callable[[str], Dict[str, object]]] = None


def natural_sort_key(path: str):
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", path)
    ]


def collect_images(folder_path: str) -> List[str]:
    image_paths: List[str] = []
    for root, _, files in os.walk(folder_path):
        for filename in files:
            if os.path.splitext(filename)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(os.path.join(root, filename))
    image_paths.sort(key=natural_sort_key)
    return image_paths


def load_liveness_logic() -> Tuple[Callable[[str], Dict[str, object]], Optional[Callable[[str], Dict[str, object]]]]:
    global _CHECK_LIVE_FN, _JSON_FMT_FN
    if _CHECK_LIVE_FN is None:
        import app as app_logic  # Reuse the exact liveness flow from app.py
        if hasattr(app_logic, "device_id"):
            app_logic.device_id = 0

        _CHECK_LIVE_FN = app_logic.check_live
        _JSON_FMT_FN = getattr(app_logic, "get_jsonfmt", None)
    return _CHECK_LIVE_FN, _JSON_FMT_FN


def format_result(raw_result: Dict[str, object], json_formatter: Optional[Callable[[str], Dict[str, object]]]) -> str:
    if not isinstance(raw_result, dict):
        return str(raw_result)

    if "error" in raw_result:
        return f"ERROR: {raw_result['error']}"

    answer = raw_result.get("face_liveness")
    if answer is None:
        return json.dumps(raw_result, indent=2, ensure_ascii=False)

    if json_formatter is None:
        return str(answer)

    try:
        return json.dumps(json_formatter(str(answer)), indent=2, ensure_ascii=False)
    except Exception:
        return str(answer)


class ResultPanel(ttk.LabelFrame):
    def __init__(self, parent: tk.Widget, title: str):
        super().__init__(parent, text=title)

        body = ttk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(0, weight=3)
        body.rowconfigure(1, weight=0)
        body.rowconfigure(2, weight=2)

        self.image_label = ttk.Label(
            body,
            text="No image",
            anchor="center",
            relief=tk.SOLID,
        )
        self.image_label.grid(row=0, column=0, sticky="nsew", pady=(0, 8))
        self.image_label.bind("<Configure>", self._on_image_resize)

        self.meta_var = tk.StringVar(value="")
        self.meta_label = ttk.Label(
            body,
            textvariable=self.meta_var,
            anchor="w",
            justify=tk.LEFT,
        )
        self.meta_label.grid(row=1, column=0, sticky="ew", pady=(0, 6))

        self.result_text = tk.Text(body, wrap=tk.WORD)
        self.result_text.grid(row=2, column=0, sticky="nsew")
        self.result_text.configure(state=tk.DISABLED)

        self._photo: Optional[ImageTk.PhotoImage] = None
        self._orig_image: Optional[Image.Image] = None

    def _set_text(self, text: str) -> None:
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, text or "")
        self.result_text.configure(state=tk.DISABLED)

    def _render_image(self) -> None:
        if self._orig_image is None:
            self._photo = None
            self.image_label.configure(image="", text="No image")
            return

        available_w = max(self.image_label.winfo_width() - 8, 80)
        available_h = max(self.image_label.winfo_height() - 8, 80)

        img = self._orig_image.copy()
        resampling = getattr(Image, "Resampling", Image)
        img.thumbnail((available_w, available_h), resampling.LANCZOS)

        self._photo = ImageTk.PhotoImage(img)
        self.image_label.configure(image=self._photo, text="")

    def _on_image_resize(self, _event: tk.Event) -> None:
        if self._orig_image is not None:
            self._render_image()

    def set_data(self, image_path: Optional[str], result: str, meta: str = "") -> None:
        if image_path:
            try:
                with Image.open(image_path) as img:
                    self._orig_image = img.convert("RGB")
                self._render_image()
            except Exception:
                self._orig_image = None
                self._photo = None
                self.image_label.configure(image="", text=f"Failed to load image\n{image_path}")
        else:
            self._orig_image = None
            self._photo = None
            self.image_label.configure(image="", text="No image")

        self.meta_var.set(meta or "")
        self._set_text(result)


class AutoCheckingApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Auto Checking Face Liveness")
        self.aspect_width = 1500
        self.aspect_height = 820
        default_width = 1200
        default_height = int(default_width * self.aspect_height / self.aspect_width)
        self.root.geometry(f"{default_width}x{default_height}")
        self.root.state("normal")
        self.root.resizable(True, True)
        self.root.minsize(900, int(900 * self.aspect_height / self.aspect_width))
        self.root.wm_aspect(
            self.aspect_width,
            self.aspect_height,
            self.aspect_width,
            self.aspect_height,
        )

        self.folder_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready")

        self._events: "queue.Queue[Tuple]" = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._stop_requested = threading.Event()

        self._build_ui()

    def _build_ui(self) -> None:
        ctrl = ttk.Frame(self.root)
        ctrl.pack(fill=tk.X, padx=10, pady=(10, 6))

        ttk.Label(ctrl, text="Folder:").pack(side=tk.LEFT)

        folder_entry = ttk.Entry(ctrl, textvariable=self.folder_var)
        folder_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        browse_button = ttk.Button(ctrl, text="Browse", command=self._browse_folder)
        browse_button.pack(side=tk.LEFT, padx=(0, 8))

        self.start_button = ttk.Button(ctrl, text="Start", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT)

        self.stop_button = ttk.Button(ctrl, text="Stop", command=self.stop_processing)
        self.stop_button.pack(side=tk.LEFT, padx=(8, 0))
        self.stop_button.state(["disabled"])

        status_row = ttk.Frame(self.root)
        status_row.pack(fill=tk.X, padx=10, pady=(0, 8))
        ttk.Label(status_row, text="Status:").pack(side=tk.LEFT)
        ttk.Label(status_row, textvariable=self.status_var).pack(side=tk.LEFT, padx=(6, 0))

        content = ttk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        content.columnconfigure(0, weight=1, uniform="panel")
        content.columnconfigure(1, weight=1, uniform="panel")
        content.rowconfigure(0, weight=1)

        self.previous_panel = ResultPanel(content, "Previous Processed")
        self.previous_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        self.current_panel = ResultPanel(content, "Current Processing")
        self.current_panel.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

    def _browse_folder(self) -> None:
        chosen = filedialog.askdirectory()
        if chosen:
            self.folder_var.set(chosen)

    def start_processing(self) -> None:
        if self._running:
            return

        folder_path = self.folder_var.get().strip()
        if not folder_path:
            self.status_var.set("Please provide a folder path")
            return

        folder_path = os.path.abspath(folder_path)
        if not os.path.isdir(folder_path):
            self.status_var.set(f"Folder not found: {folder_path}")
            return

        image_paths = collect_images(folder_path)
        if not image_paths:
            self.status_var.set(f"No images found in: {folder_path}")
            return

        self.previous_panel.set_data(None, "", "")
        self.current_panel.set_data(None, "", "")

        self._running = True
        self._stop_requested.clear()
        self.start_button.state(["disabled"])
        self.stop_button.state(["!disabled"])
        self.status_var.set(f"Found {len(image_paths)} images. Loading liveness model...")

        self._worker = threading.Thread(
            target=self._process_images_worker,
            args=(image_paths,),
            daemon=True,
        )
        self._worker.start()
        self.root.after(100, self._poll_events)

    def stop_processing(self) -> None:
        if not self._running:
            return

        self._stop_requested.set()
        self.stop_button.state(["disabled"])
        self.status_var.set("Stop requested. Finishing current image...")

    def _process_images_worker(self, image_paths: List[str]) -> None:
        total = len(image_paths)

        try:
            check_live, json_formatter = load_liveness_logic()
        except Exception as exc:
            self._events.put(("status", f"Failed to load app.py logic: {exc}"))
            self._events.put(("done",))
            return

        processed_count = 0
        for index, image_path in enumerate(image_paths, start=1):
            if self._stop_requested.is_set():
                break

            self._events.put(
                (
                    "current",
                    image_path,
                    f"Processing {index}/{total}\n{image_path}",
                    f"Path: {image_path}",
                    f"Running {index}/{total}",
                )
            )

            start_time = time.time()
            try:
                raw_result = check_live(image_path)
                result_text = format_result(raw_result, json_formatter)
            except Exception as exc:
                result_text = f"ERROR: {exc}"
            elapsed = time.time() - start_time

            processed_count += 1
            self._events.put(
                (
                    "previous",
                    image_path,
                    result_text,
                    f"Path: {image_path}\nProcessing time: {elapsed:.3f} sec",
                    f"Completed {index}/{total}",
                )
            )
            self._events.put(("clear_current",))

        if self._stop_requested.is_set():
            self._events.put(("status", f"Stopped. Processed {processed_count}/{total} images."))
        else:
            self._events.put(("status", f"Done. Processed {processed_count}/{total} images."))
        self._events.put(("done",))

    def _poll_events(self) -> None:
        while True:
            try:
                event = self._events.get_nowait()
            except queue.Empty:
                break

            event_type = event[0]
            if event_type == "current":
                _, image_path, text, meta, status = event
                self.current_panel.set_data(image_path, text, meta)
                self.status_var.set(status)
            elif event_type == "previous":
                _, image_path, text, meta, status = event
                self.previous_panel.set_data(image_path, text, meta)
                self.status_var.set(status)
            elif event_type == "clear_current":
                self.current_panel.set_data(None, "", "")
            elif event_type == "status":
                self.status_var.set(event[1])
            elif event_type == "done":
                self._running = False
                self.start_button.state(["!disabled"])
                self.stop_button.state(["disabled"])

        if self._running:
            self.root.after(100, self._poll_events)


def main() -> None:
    root = tk.Tk()
    AutoCheckingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
