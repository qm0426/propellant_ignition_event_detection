import csv
import re
import shutil
from datetime import datetime
from pathlib import Path
import sys

# Optional sensitivity workflow uses a writable local module cache. This branch
# does not run for the original detection command or when importing this module.
if __name__ == "__main__" and sys.argv[1:2] == ["--prompt-sensitivity"]:
    import os
    _prompt_suite = Path(__file__).resolve().parent
    os.environ.setdefault("HF_MODULES_CACHE", str(_prompt_suite / "revision_outputs" / "prompt_runtime_cache"))

import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer, AutoProcessor

"""
LocateAnything demo — batch two-state aluminum droplet detection.

Default (all images in test_images/):
    python test_locate_anything.py

Custom output folder:
    python test_locate_anything.py --image-dir test_images --output-dir locate_results

Single image + show:
    python test_locate_anything.py --image test_images\\MCFF0004-N001_00004355.tif --show

Single-prompt mode:
    python test_locate_anything.py --no-two-state --prompt "small bright molten aluminum droplets"

Two-state multi-prompt example:
    python test_locate_anything.py ^
      --image test_images\\MCFF0004-N001_00004355.tif ^
      --two-state ^
      --generation-mode hybrid ^
      --temperature 0.2 ^
      --nms-iou 0.5 ^
      --show
"""

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
BOX_COLOR_MAP = {
    "unburned_aluminum_droplet": "cyan",
    "burning_aluminum_droplet": "red",
}

LABEL_SHORT = {
    "unburned_aluminum_droplet": "unburned",
    "burning_aluminum_droplet": "burning",
}

# DEFAULT_TWO_STATE_PROMPTS = {
#     "unburned_aluminum_droplet": [
#         "small dark round particles",
#         "tiny black circular particles",
#         "cold non glowing spherical droplets",
#         "small gray round droplets",
#         "small non glowing round particles",
#     ],
#     "burning_aluminum_droplet": [
#         "small bright orange glowing droplets",
#         "hot glowing metal droplets",
#         "small bright spherical droplets",
#         "tiny faint orange glowing spots",
#         "newly ignited small orange particles",
#         "weak incandescent orange specks",
#         "small dim glowing orange dots",
#         "small orange dots that have just started glowing",
#     ],
# }
DEFAULT_TWO_STATE_PROMPTS = {
    "unburned_aluminum_droplet": [
        "tiny dark round particles",
        "small black circular droplets",
        "small gray non glowing round particles",
        "isolated compact dark specks",
        "small non glowing spherical particles",
    ],
    "burning_aluminum_droplet": [
        "compact round orange glowing droplets",
        "small isolated bright glowing particles",
        "small bright spherical droplets",
        "tiny faint orange glowing dots",
        "newly ignited small orange particles",
        "small round incandescent specks",
        "small dim glowing orange dots",
    ],
}

def ensure_prompt_list(prompts):
    if isinstance(prompts, str):
        return [prompts]
    if isinstance(prompts, (list, tuple)):
        return list(prompts)
    raise TypeError(f"Unsupported prompt type: {type(prompts)}")


def box_iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a["x2"] - a["x1"]) * max(0.0, a["y2"] - a["y1"])
    area_b = max(0.0, b["x2"] - b["x1"]) * max(0.0, b["y2"] - b["y1"])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def nms_boxes(boxes, iou_threshold=0.5, class_aware=True):
    if not boxes:
        return []

    indexed = []
    for i, b in enumerate(boxes):
        bb = dict(b)
        bb.setdefault("score", 1.0)
        bb["_order"] = i
        indexed.append(bb)

    kept = _nms_boxes_indexed(indexed, iou_threshold=iou_threshold, class_aware=class_aware)
    for b in kept:
        b.pop("_order", None)
    return kept


def _nms_boxes_indexed(indexed, iou_threshold=0.5, class_aware=True):
    if class_aware:
        labels = sorted({b.get("label", "") for b in indexed})
        kept = []
        for label in labels:
            group = [b for b in indexed if b.get("label", "") == label]
            kept.extend(_nms_boxes_indexed(group, iou_threshold=iou_threshold, class_aware=False))
        kept.sort(key=lambda x: x["_order"])
        return kept

    indexed.sort(key=lambda x: (-x.get("score", 1.0), x["_order"]))
    kept = []
    while indexed:
        best = indexed.pop(0)
        kept.append(best)
        remain = []
        for cand in indexed:
            if box_iou(best, cand) < iou_threshold:
                remain.append(cand)
        indexed = remain

    kept.sort(key=lambda x: x["_order"])
    return kept


def normalize_box(box, image_width=None, image_height=None):
    """确保 x1<=x2, y1<=y2，并可选裁剪到图像范围内。"""
    b = dict(box)
    x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if image_width is not None:
        x1 = max(0.0, min(x1, image_width))
        x2 = max(0.0, min(x2, image_width))
    if image_height is not None:
        y1 = max(0.0, min(y1, image_height))
        y2 = max(0.0, min(y2, image_height))
    b["x1"], b["y1"], b["x2"], b["y2"] = x1, y1, x2, y2
    return b


class LocateAnythingWorker:
    def __init__(
        self,
        model_path="nvidia/LocateAnything-3B",
        device="cuda",
        dtype=torch.bfloat16,
    ):
        self.device = device
        self.dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        self.model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=dtype,
        ).to(device).eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode="hybrid",
        max_new_tokens=2048,
        temperature=0.2,
        verbose=True,
    ):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": question},
                ],
            }
        ]

        text = self.processor.py_apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        images, videos = self.processor.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,  # "fast", "hybrid", "slow"
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        answer = response[0] if isinstance(response, tuple) else response
        return answer

    def detect(self, image: Image.Image, categories, **kwargs):
        """
        开放词汇多类别检测。
        categories: ["person", "car"] 这种列表。
        """
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs):
        """
        自然语言短语定位多个实例。
        对你的铝液滴任务，优先用这个。
        """
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_multi_class(self, image: Image.Image, class_prompts: dict, **kwargs):
        """
        按 label -> phrase(s) 逐类、逐 prompt 调用 ground_multi，合并 boxes。
        兼容 str 或 list/tuple prompt。
        """
        all_boxes = []
        w, h = image.size

        for label, prompts in class_prompts.items():
            prompt_list = ensure_prompt_list(prompts)
            for prompt_idx, phrase in enumerate(prompt_list):
                print(f"\nDetecting class: {label}")
                print(f"Prompt {prompt_idx + 1}/{len(prompt_list)}: {phrase}")

                answer = self.ground_multi(image, phrase, **kwargs)
                print(f"Raw answer for {label} / prompt {prompt_idx + 1}:")
                print(answer)

                boxes = self.parse_boxes(answer, w, h)
                print(f"Parsed {len(boxes)} boxes for {label} / prompt {prompt_idx + 1}")

                for b in boxes:
                    b["label"] = label
                    b["prompt"] = phrase
                    b["prompt_index"] = prompt_idx
                    all_boxes.append(b)

        return all_boxes

    def detect_two_state_aluminum_droplets(
        self,
        image: Image.Image,
        prompts_by_class: dict | None = None,
        class_prompts: dict | None = None,
        **kwargs,
    ):
        """两状态铝液滴检测；每个类别下多个 prompt 分别检测后合并。"""
        if prompts_by_class is None:
            prompts_by_class = class_prompts
        if prompts_by_class is None:
            prompts_by_class = DEFAULT_TWO_STATE_PROMPTS

        all_boxes = []
        w, h = image.size

        for label, prompts in prompts_by_class.items():
            prompt_list = ensure_prompt_list(prompts)

            for prompt_idx, phrase in enumerate(prompt_list):
                print(f"\nDetecting class: {label}")
                print(f"Prompt {prompt_idx + 1}/{len(prompt_list)}: {phrase}")

                answer = self.ground_multi(image, phrase, **kwargs)
                print(f"Raw answer for {label} / prompt {prompt_idx + 1}:")
                print(answer)

                boxes = self.parse_boxes(answer, w, h)
                print(f"Parsed {len(boxes)} boxes for {label} / prompt {prompt_idx + 1}")

                for b in boxes:
                    b["label"] = label
                    b["prompt"] = phrase
                    b["prompt_index"] = prompt_idx
                    all_boxes.append(b)

        return all_boxes

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int):
        """
        LocateAnything 输出坐标是 0-1000 归一化坐标。
        这里转成原图像素坐标。
        """
        boxes = []
        pattern = r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>"

        for m in re.finditer(pattern, answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append(
                normalize_box(
                    {
                        "x1": x1 / 1000 * image_width,
                        "y1": y1 / 1000 * image_height,
                        "x2": x2 / 1000 * image_width,
                        "y2": y2 / 1000 * image_height,
                    },
                    image_width=image_width,
                    image_height=image_height,
                )
            )

        return boxes


def is_image_file(path) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def list_images(image_dir: str):
    paths = []
    for path in sorted(Path(image_dir).iterdir()):
        if path.is_file() and is_image_file(path):
            paths.append(str(path))
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def collect_image_paths(image: str | None = None, image_dir: str = "test_images"):
    """收集待处理图片路径；按后缀过滤，只保留图片文件。"""
    if image is not None:
        path = Path(image)
        if path.is_dir():
            return list_images(str(path))
        if not path.is_file():
            raise FileNotFoundError(f"Image path not found: {image}")
        if not is_image_file(path):
            raise ValueError(
                f"Unsupported file suffix '{path.suffix}' for {image}. "
                f"Supported: {', '.join(sorted(IMAGE_EXTENSIONS))}"
            )
        return [str(path)]
    return list_images(image_dir)


def load_rgb_image(image_path: str) -> Image.Image:
    if not is_image_file(image_path):
        raise ValueError(f"Not an image file: {image_path}")
    return Image.open(image_path).convert("RGB")


def save_boxes_txt(boxes, txt_path: str):
    """保存为 tab 分隔：label, x1, y1, x2, y2"""
    lines = ["label\tx1\ty1\tx2\ty2"]
    for b in boxes:
        label = b.get("label", "")
        lines.append(
            f"{label}\t{b['x1']:.2f}\t{b['y1']:.2f}\t{b['x2']:.2f}\t{b['y2']:.2f}"
        )
    Path(txt_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved TXT: {txt_path}")


def load_boxes_txt(txt_path: str):
    """读取 save_boxes_txt 保存的结果。"""
    boxes = []
    with open(txt_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            boxes.append(
                {
                    "label": row["label"],
                    "x1": float(row["x1"]),
                    "y1": float(row["y1"]),
                    "x2": float(row["x2"]),
                    "y2": float(row["y2"]),
                }
            )
    return boxes


def detect_boxes_for_image(
    worker: LocateAnythingWorker,
    image: Image.Image,
    two_state: bool,
    class_prompts: dict,
    single_prompt: str,
    gen_kwargs: dict,
    nms_iou: float = 0.5,
):
    if two_state:
        boxes = worker.detect_two_state_aluminum_droplets(
            image,
            prompts_by_class=class_prompts,
            **gen_kwargs,
        )
        count_before = len(boxes)
        if nms_iou >= 0:
            boxes = nms_boxes(boxes, iou_threshold=nms_iou, class_aware=True)
        print(f"\nTotal boxes before NMS: {count_before}")
        print(f"Total boxes after NMS: {len(boxes)}")
        return boxes

    answer = worker.ground_multi(image, single_prompt, **gen_kwargs)
    print("Raw answer:")
    print(answer)
    w, h = image.size
    boxes = worker.parse_boxes(answer, w, h)
    for b in boxes:
        b["label"] = single_prompt
        b["prompt"] = single_prompt
        b["prompt_index"] = 0
    return boxes


def draw_boxes(
    image: Image.Image,
    boxes,
    output_path="locate_result.png",
    show: bool = False,
    block: bool = True,
):
    vis = image.copy()
    draw = ImageDraw.Draw(vis)
    img_w, img_h = image.size

    for i, b in enumerate(boxes):
        b = normalize_box(b, image_width=img_w, image_height=img_h)
        x1, y1, x2, y2 = b["x1"], b["y1"], b["x2"], b["y2"]
        label = b.get("label", "unknown")
        color = BOX_COLOR_MAP.get(label, "yellow")
        short = LABEL_SHORT.get(label, label)
        caption = f"{i}: {short}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, max(0, y1 - 15)), caption, fill=color)

    vis.save(output_path)
    print(f"Saved: {output_path}")

    if show:
        _show_detection_popup(vis, title=output_path, block=block)

    return vis


def save_boxes_csv(boxes, csv_path: str, image_path: str = ""):
    fieldnames = ["image_path", "label", "prompt", "prompt_index", "x1", "y1", "x2", "y2"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for b in boxes:
            writer.writerow(
                {
                    "image_path": image_path,
                    "label": b.get("label", ""),
                    "prompt": b.get("prompt", ""),
                    "prompt_index": b.get("prompt_index", ""),
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                }
            )
    print(f"Saved CSV: {csv_path}")


def print_boxes_summary(boxes):
    print(f"\nTotal boxes: {len(boxes)}")
    if not boxes:
        print("No boxes detected.")
        return
    for i, b in enumerate(boxes):
        print(
            f"{i}, {b.get('label', 'unknown')}, {b.get('prompt_index', '')}, "
            f"{b.get('prompt', '')}, "
            f"{b['x1']:.1f}, {b['y1']:.1f}, {b['x2']:.1f}, {b['y2']:.1f}"
        )


def _show_detection_popup(image: Image.Image, title: str = "detection", block: bool = True):
    """弹窗显示检测结果；保存文件与弹窗使用同一张 vis 图。"""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(title)
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=block)
    except Exception as exc:
        print(f"matplotlib 弹窗失败 ({exc})，改用 PIL.show()")
        image.show(title=title)


def make_default_output_dir(base_name: str = "locate_results") -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(Path(base_name) / f"run_{stamp}")


def process_image_paths(
    worker,
    image_paths,
    output_dir: str,
    two_state: bool,
    class_prompts: dict,
    single_prompt: str,
    gen_kwargs: dict,
    nms_iou: float = 0.5,
    show: bool = False,
    block: bool = True,
    save_aggregate_csv: bool = True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    txt_dir = output_dir / "detections_txt"
    vis_dir = output_dir / "vis"
    postprocess_dir = output_dir / "postprocessed"
    txt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    postprocess_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.resolve()}")
    print(f"  detections_txt: {txt_dir.resolve()}")
    print(f"  vis: {vis_dir.resolve()}")
    print(f"  postprocessed: {postprocess_dir.resolve()} (empty, for postprocess output)")

    all_rows = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"\n{'=' * 60}")
        print(f"[{idx}/{len(image_paths)}] {image_path}")

        image = load_rgb_image(image_path)
        boxes = detect_boxes_for_image(
            worker,
            image,
            two_state=two_state,
            class_prompts=class_prompts,
            single_prompt=single_prompt,
            gen_kwargs=gen_kwargs,
            nms_iou=nms_iou,
        )
        print_boxes_summary(boxes)

        stem = Path(image_path).stem
        vis_path = vis_dir / f"{stem}.png"
        txt_path = txt_dir / f"{stem}.txt"

        draw_boxes(
            image,
            boxes,
            output_path=str(vis_path),
            show=show and idx == len(image_paths),
            block=block,
        )
        save_boxes_txt(boxes, str(txt_path))

        for b in boxes:
            all_rows.append(
                {
                    "image_path": image_path,
                    "label": b.get("label", ""),
                    "prompt": b.get("prompt", ""),
                    "prompt_index": b.get("prompt_index", ""),
                    "x1": b["x1"],
                    "y1": b["y1"],
                    "x2": b["x2"],
                    "y2": b["y2"],
                }
            )

    if save_aggregate_csv:
        csv_path = output_dir / "all_detections.csv"
        fieldnames = ["image_path", "label", "prompt", "prompt_index", "x1", "y1", "x2", "y2"]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nSaved aggregate CSV: {csv_path}")

    print(f"\nDone. Processed {len(image_paths)} image(s).")
    return all_rows


if __name__ == "__main__":    
    import argparse
    if sys.argv[1:2] == ["--prompt-sensitivity"]:
        _prompt_suite = Path(__file__).resolve().parent
        if not (_prompt_suite / "prompt_sensitivity_tools" / "runner.py").is_file():
            raise FileNotFoundError("Missing prompt sensitivity tools: " + str(_prompt_suite))
        sys.path.insert(0, str(_prompt_suite))
        from prompt_sensitivity_tools.runner import main as _prompt_sensitivity_main
        _prompt_sensitivity_main(sys.argv[2:])
        raise SystemExit(0)
    parser = argparse.ArgumentParser(description="LocateAnything open-vocabulary detection demo",
        epilog="Prompt wording validation: python test_locate_anything.py --prompt-sensitivity all --device cuda")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="test_images",
        help="Directory of images to process (default: test_images)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional image file or directory; only image suffixes are accepted",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output folder for vis images and txt results (default: locate_results/run_YYYYMMDD_HHMMSS)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="small bright molten aluminum droplets",
        help="Natural language phrase for single-prompt mode (--no-two-state)",
    )
    parser.add_argument("--model", type=str, default="nvidia/LocateAnything-3B")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--generation-mode", type=str, default="hybrid", choices=["fast", "hybrid", "slow"])
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print model verbose generation logs",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="弹窗显示最后一张检测图",
    )
    parser.add_argument(
        "--no-block",
        action="store_true",
        help="弹窗不阻塞（仅 --show 时有效）",
    )
    parser.add_argument(
        "--two-state",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Detect unburned and burning aluminum droplets separately (default: on)",
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="IoU threshold for class-aware NMS in two-state mode. Set negative to disable NMS.",
    )
    parser.add_argument(
        "--prompt-unburned",
        type=str,
        default=None,
        help="Optional single override prompt for unburned_aluminum_droplet",
    )
    parser.add_argument(
        "--prompt-burning",
        type=str,
        default=None,
        help="Optional single override prompt for burning_aluminum_droplet",
    )
    parser.add_argument(
        "--save-csv",
        type=str,
        default=None,
        help="Optional extra CSV path (aggregate CSV is always saved in output-dir)",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or make_default_output_dir()

    if args.image:
        image_paths = collect_image_paths(image=args.image, image_dir=args.image_dir)
    else:
        image_paths = collect_image_paths(image_dir=args.image_dir)

    worker = LocateAnythingWorker(
        model_path=args.model,
        device=args.device,
        dtype=torch.bfloat16,
    )

    gen_kwargs = dict(
        generation_mode=args.generation_mode,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        verbose=args.verbose,
    )

    class_prompts = {
        "unburned_aluminum_droplet": (
            args.prompt_unburned
            if args.prompt_unburned is not None
            else DEFAULT_TWO_STATE_PROMPTS["unburned_aluminum_droplet"]
        ),
        "burning_aluminum_droplet": (
            args.prompt_burning
            if args.prompt_burning is not None
            else DEFAULT_TWO_STATE_PROMPTS["burning_aluminum_droplet"]
        ),
    }

    mode = "two-state" if args.two_state else "single-prompt"
    print(f"Mode: {mode}")
    print(f"Images: {len(image_paths)}")
    if args.two_state:
        for label, prompts in class_prompts.items():
            prompt_list = ensure_prompt_list(prompts)
            print(f"  {label}: {len(prompt_list)} prompt(s)")

    process_image_paths(
        worker=worker,
        image_paths=image_paths,
        output_dir=output_dir,
        two_state=args.two_state,
        class_prompts=class_prompts,
        single_prompt=args.prompt,
        gen_kwargs=gen_kwargs,
        nms_iou=args.nms_iou,
        show=args.show,
        block=not args.no_block,
        save_aggregate_csv=True,
    )

    if args.save_csv:
        aggregate = Path(output_dir) / "all_detections.csv"
        if aggregate.exists():
            shutil.copyfile(aggregate, args.save_csv)
            print(f"Copied aggregate CSV to: {args.save_csv}")
