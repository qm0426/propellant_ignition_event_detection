
import re



from pathlib import Path


import torch

from PIL import Image

from transformers import AutoModel, AutoTokenizer, AutoProcessor

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

BOX_COLOR_MAP = {
    "unburned_aluminum_droplet": "cyan",
    "burning_aluminum_droplet": "red",
}

LABEL_SHORT = {
    "unburned_aluminum_droplet": "unburned",
    "burning_aluminum_droplet": "burning",
}

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
