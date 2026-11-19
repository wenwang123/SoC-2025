#!/usr/bin/env python3
"""Single-entry RTL OCR test using an existing PaddleOCR source checkout.

Only custom evaluation code is bundled. Official PaddleOCR modules, exported
models and test images are supplied separately through command-line paths.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import ctypes.util
import hashlib
import html
import json
import logging
import random
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont, features
from rapidfuzz.distance import Levenshtein

FONT = None
LANGUAGES = {"arabic": "029_ara_阿拉伯语", "hebrew": "028_heb_希伯来语"}
SOURCES = ["images_and_label_1019", "scan_images"]
# Both image variants of these documents were used in this model's training.
TRAINING_DOCUMENTS = {
    "77214", "90398", "90931",
    "37364_fc2d66d3110043d9b3ad473e32f1e9cc_02",
    "43954_fb85b73b06724142892dc2d7564c3439_01",
    "11721_07be4cb910ab4b40be3afe69ad3fef5d_01",
}


class VisualOrder:
    """Use FriBidi as Pillow/RAQM does, retaining ordinary Unicode letters.

    Obtain index maps without Arabic presentation-form shaping. Apply glyph
    mirroring to punctuation; keep combining marks with their base characters
    using FriBidi's default NSM reordering. Do not normalize/spell-correct text.
    """

    def __init__(self):
        library = ctypes.util.find_library("fribidi")
        if not library:
            raise RuntimeError("Pillow-compatible libfribidi is required")
        self.lib = ctypes.CDLL(library)
        self.char = ctypes.c_uint32
        self.index = ctypes.c_int
        self.level = ctypes.c_uint8
        pointer = ctypes.POINTER
        self.lib.fribidi_get_bidi_type.argtypes = [self.char]
        self.lib.fribidi_get_bidi_type.restype = self.char
        self.lib.fribidi_log2vis.argtypes = [
            pointer(self.char), self.index, pointer(self.char),
            pointer(self.char), pointer(self.index), pointer(self.index),
            pointer(self.level),
        ]
        self.lib.fribidi_log2vis.restype = self.level
        self.lib.fribidi_get_mirror_char.argtypes = [self.char, pointer(self.char)]
        self.lib.fribidi_get_mirror_char.restype = self.index
        self.version = ctypes.c_char_p.in_dll(
            self.lib, "fribidi_version_info"
        ).value.decode().splitlines()[0]

    def __call__(self, text, base=None):
        if not text:
            raise ValueError("Empty text")
        size = len(text)
        logical = (self.char * size)(*map(ord, text))
        visual_to_logical = (self.index * size)()
        levels = (self.level * size)()
        representative = {None: ".", "L": "A", "R": "א"}[base]
        direction = self.char(self.lib.fribidi_get_bidi_type(ord(representative)))
        result = self.lib.fribidi_log2vis(
            logical, size, ctypes.byref(direction), None, None,
            visual_to_logical, levels,
        )
        if not result:
            raise RuntimeError(f"FriBidi failed for {text!r}")
        output = []
        for index in visual_to_logical:
            char = ord(text[index])
            mirror = self.char()
            if levels[index] % 2 and self.lib.fribidi_get_mirror_char(
                char, ctypes.byref(mirror)
            ):
                char = mirror.value
            output.append(chr(char))
        return "".join(output)


def normalize(text, keep_spaces=True):
    text = unicodedata.normalize("NFC", text or "")
    text = text.replace("\ufeff", "")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    if keep_spaces:
        text = re.sub(r"\s+", " ", text).strip()
    else:
        text = "".join(ch for ch in text if not ch.isspace())
    return text


def score(pred, truth):
    pred_space, truth_space = normalize(pred), normalize(truth)
    pred_no_space, truth_no_space = normalize(pred, False), normalize(truth, False)
    distance = Levenshtein.distance(pred_space, truth_space)
    distance_no_space = Levenshtein.distance(pred_no_space, truth_no_space)
    return {
        "ground_truth_chars": len(truth_space),
        "prediction_chars": len(pred_space),
        "edit_distance": distance,
        "cer": distance / len(truth_space) if truth_space else (0.0 if not pred_space else 1.0),
        "ground_truth_chars_no_space": len(truth_no_space),
        "prediction_chars_no_space": len(pred_no_space),
        "edit_distance_no_space": distance_no_space,
        "cer_no_space": distance_no_space / len(truth_no_space) if truth_no_space else (0.0 if not pred_no_space else 1.0),
    }


@lru_cache(maxsize=256)
def font(size):
    return ImageFont.truetype(FONT, size=max(1, int(size)))


def text_patch(text, width, height):
    """Fit actual glyph bounds, including baseline/diacritics, inside the box."""
    width, height = max(2, width), max(2, height)
    tile = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(tile)
    size = max(2, height)
    padding = min(2, height // 8)
    for _ in range(8):
        face = font(size)
        bbox = draw.textbbox((0, 0), text, font=face, direction='ltr')
        tw, th = max(1, bbox[2]-bbox[0]), max(1, bbox[3]-bbox[1])
        scale = min((width-2*padding)/tw, (height-2*padding)/th, 1)
        if scale >= 0.99 or size <= 1:
            break
        size = max(1, int(size*scale)-1)
    x = width-padding-tw-bbox[0]
    y = (height-th)//2-bbox[1]
    draw.text((x, y), text, font=face, fill='black', direction='ltr')
    return np.asarray(tile)


def add_cer_header(picture, cer):
    """Add a page CER above the existing spatial visualization."""
    width, height = picture.size
    face = font(max(20, min(64, width // 65)))
    text = f'CER: {cer:.2%}'
    bbox = face.getbbox(text)
    margin = max(10, face.size // 4)
    header_height = bbox[3] - bbox[1] + 2 * margin
    result = Image.new('RGB', (width, height + header_height), 'white')
    result.paste(picture, (0, header_height))
    ImageDraw.Draw(result).text((margin, margin-bbox[1]), text, font=face, fill='black')
    return result


def render(source, rows, cer=None):
    rng = random.Random(0)
    original = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    overlay = original.copy()
    right = np.full_like(original, 255)
    height, width = original.shape[:2]
    for entry in rows:
        quad = np.asarray(entry['points'], dtype=np.float32)
        color = tuple(rng.randint(35, 210) for _ in range(3))
        cv2.fillPoly(overlay, [quad.astype(np.int32)], color)
        bw = max(2, round(float(np.linalg.norm(quad[1]-quad[0]))))
        bh = max(2, round(float(np.linalg.norm(quad[3]-quad[0]))))
        patch = text_patch(entry['transcription_logical'], bw, bh)
        x0 = max(0, int(np.floor(quad[:, 0].min()))-1)
        y0 = max(0, int(np.floor(quad[:, 1].min()))-1)
        x1 = min(width, int(np.ceil(quad[:, 0].max()))+2)
        y1 = min(height, int(np.ceil(quad[:, 1].max()))+2)
        if x1<=x0 or y1<=y0:
            continue
        local = quad - [x0, y0]
        transform = cv2.getPerspectiveTransform(
            np.asarray([[0, 0], [bw-1, 0], [bw-1, bh-1], [0, bh-1]], dtype=np.float32),
            local.astype(np.float32))
        fitted = cv2.warpPerspective(patch, transform, (x1-x0, y1-y0), borderValue=(255,255,255))
        roi = right[y0:y1, x0:x1]
        np.minimum(roi, fitted, out=roi)
        cv2.polylines(right, [quad.astype(np.int32)], True, color, 1)
    left = cv2.addWeighted(original, 0.65, overlay, 0.35, 0)
    picture = Image.fromarray(np.concatenate([left, right], axis=1))
    return add_cer_header(picture, cer) if cer is not None else picture


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def summarize(rows):
    chars = sum(row["ground_truth_chars"] for row in rows)
    edits = sum(row["edit_distance"] for row in rows)
    chars_ns = sum(row["ground_truth_chars_no_space"] for row in rows)
    edits_ns = sum(row["edit_distance_no_space"] for row in rows)
    return {"images": len(rows), "documents": len({(r["language"], r["stem"]) for r in rows}),
            "ground_truth_chars": chars, "edit_distance": edits,
            "cer": edits / chars if chars else None, "cer_no_space": edits_ns / chars_ns if chars_ns else None,
            "exact_images": sum(r["edit_distance"] == 0 for r in rows), "boxes": sum(r["boxes"] for r in rows)}


def validate_model(directory):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for name in ["inference.yml", "inference.pdiparams"]:
        if not (directory / name).is_file():
            raise FileNotFoundError(directory / name)
    if not any((directory / name).is_file() for name in ["inference.json", "inference.pdmodel"]):
        raise FileNotFoundError(f"No inference.json / inference.pdmodel in {directory}")


def choose_font(explicit):
    if explicit:
        path = explicit.resolve()
        ImageFont.truetype(str(path), 16)
        return str(path)
    try:
        return str(ImageFont.truetype("DejaVuSans.ttf", 16).path)
    except OSError as exc:
        raise RuntimeError("找不到 DejaVuSans.ttf，请用 --font 指定支持阿拉伯语和希伯来语的字体") from exc


def collect_inputs(args):
    rows = []
    for language, folder in LANGUAGES.items():
        labels = args.test_data / "images_and_label_1019" / folder
        for source in SOURCES:
            directory = args.test_data / source / folder
            if not directory.is_dir() or not labels.is_dir():
                raise FileNotFoundError(f"Missing test directory: {directory} or {labels}")
            images = sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})
            if not images:
                raise ValueError(f"No test images: {directory}")
            if len({p.stem for p in images}) != len(images):
                raise ValueError(f"Duplicate image stems: {directory}")
            if args.limit_per_group:
                images = images[:args.limit_per_group]
            for path in images:
                label = labels / (path.stem + ".txt")
                if not label.is_file():
                    raise FileNotFoundError(label)
                truth = label.read_text(encoding="utf8")
                if not normalize(truth):
                    raise ValueError(f"Empty ground truth: {label}")
                rows.append({"language": language, "source": source,
                             "group": language + ("_scan" if source == "scan_images" else "_images_and_label_1019"),
                             "image": path.name, "stem": path.stem, "image_path": str(path), "label_path": str(label),
                             "image_sha256": digest(path), "label_sha256": digest(label),
                             "in_primary_test": path.stem not in TRAINING_DOCUMENTS})
    if not any(row["in_primary_test"] for row in rows):
        raise ValueError("No primary-test images after training-document exclusions")
    return rows


def create_system(args, dictionary):
    # These are the only external project imports: use the official checkout.
    sys.path.insert(0, str(args.paddleocr_root))
    from tools.infer import utility
    from tools.infer.predict_system import TextSystem
    from ppocr.utils.logging import get_logger

    options = utility.init_args().parse_args([])
    options.use_gpu = args.use_gpu
    for key in ["use_xpu", "use_npu", "use_mlu", "use_metax_gpu", "use_gcu"]:
        setattr(options, key, False)
    options.det_model_dir = str(args.det_model)
    options.rec_model_dir = str(args.rec_model)
    options.rec_char_dict_path = str(dictionary)
    options.rec_algorithm = "SVTR_LCNet"
    options.rec_image_shape = "3,48,320"
    options.max_text_length = 40
    options.rec_batch_num = 16
    options.det_limit_side_len = 960
    options.det_limit_type = "max"
    options.drop_score = .3
    options.det_box_type = "quad"
    options.use_angle_cls = False
    options.show_log = False
    options.ir_optim = True
    options.save_crop_res = False
    options.warmup = False
    options.cpu_threads = 4
    options.vis_font_path = FONT
    system = TextSystem(options)
    # Official code infers Arabic reversal from the dictionary path. This
    # visual-order model needs exactly one explicit conversion below, even if
    # a user places the package in a folder whose name contains "arabic".
    system.text_recognizer.postprocess_op.reverse = False
    get_logger().setLevel(logging.ERROR)
    return system


STYLE = "body{font:16px sans-serif;margin:24px;color:#182330;max-width:1600px}table{border-collapse:collapse;width:100%;margin:15px 0}td,th{border:1px solid #ddd;padding:8px;text-align:left}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(270px,1fr));gap:16px}.card{border:1px solid #ddd;padding:10px;overflow-wrap:anywhere}.card img{height:190px;object-fit:contain;width:100%}img{max-width:100%}.excluded{background:#fff0da}bdi{font-size:22px;white-space:pre-wrap}code{overflow-wrap:anywhere}"
NAMES = {"arabic": "阿拉伯语", "hebrew": "希伯来语", "images_and_label_1019": "原图", "scan_images": "扫描图"}


def visualize(args, row, image, entries):
    directory = args.output / row["group"]
    for name in ["visualized", "thumbnails", "pages", "text"]:
        (directory / name).mkdir(parents=True, exist_ok=True)
    picture = render(image, entries, cer=row["cer"])
    picture.save(directory / "visualized" / (row["stem"] + ".jpg"), quality=94)
    thumbnail = picture.copy()
    thumbnail.thumbnail((720, 400))
    thumbnail.save(directory / "thumbnails" / (row["stem"] + ".jpg"), quality=87)
    (directory / "text" / (row["stem"] + ".txt")).write_text(row["prediction"] + "\n", encoding="utf8")
    status = "主测试" if row["in_primary_test"] else "已用于训练，仅诊断，不计入主 CER"
    table = "".join(f"<tr><td>{i}</td><td><bdi dir='auto'>{html.escape(e['transcription_logical'])}</bdi></td><td>{e['score']:.3f}</td></tr>" for i, e in enumerate(entries, 1))
    page = f"<!doctype html><meta charset='utf-8'><title>{html.escape(row['image'])}</title><style>{STYLE}</style><a href='../../index.html'>返回总览</a><h1>{html.escape(row['image'])}</h1><p>{status} · CER {row['cer']:.2%} · 编辑距离 {row['edit_distance']}/{row['ground_truth_chars']}</p><p>左：输入与检测区域；右：同位置识别文字。</p><img src='../visualized/{row['stem']}.jpg'><h2>原始标注</h2><p><bdi dir='auto'>{html.escape(row['ground_truth'])}</bdi></p><h2>识别文本</h2><p><bdi dir='auto'>{html.escape(row['prediction'])}</bdi></p><table><tr><th>区域</th><th>识别文本</th><th>置信度</th></tr>{table}</table>"
    (directory / "pages" / (row["stem"] + ".html")).write_text(page, encoding="utf8")


def reports(args, rows):
    primary = [row for row in rows if row["in_primary_test"]]
    summary = []
    for language in LANGUAGES:
        for source in SOURCES:
            selected = [row for row in primary if row["language"] == language and row["source"] == source]
            if selected:
                summary.append({"language": language, "source": source, **summarize(selected)})
    result = {"scope": "End-to-end page CER; known training documents excluded from primary results",
              "metric": "micro Levenshtein CER, NFC, remove Cf, collapse whitespace, preserve punctuation/case/marks",
              "limit_per_group": args.limit_per_group, "languages": summary, "overall": summarize(primary),
              "by_source": {source: summarize([r for r in primary if r["source"] == source]) for source in SOURCES},
              "training_pages_diagnostic_only": summarize([r for r in rows if not r["in_primary_test"]]), "all_images": rows}
    write_json(args.output / "cer_report.json", result)
    keys = [k for k in rows[0] if k not in {"ground_truth", "prediction"}]
    with (args.output / "per_image_metrics.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row[k] for k in keys} for row in rows)
    lines = ["# 阿拉伯语／希伯来语测试", "", "| 语言 | 图像 | 图片数 | CER | 无空格 CER |", "|---|---|---:|---:|---:|"]
    for row in summary:
        lines.append(f"| {NAMES[row['language']]} | {NAMES[row['source']]} | {row['images']} | {row['cer']:.2%} | {row['cer_no_space']:.2%} |")
    lines += ["", f"主结果：{result['overall']['documents']} 份文档、{len(primary)} 张图片变体，总体 CER **{result['overall']['cer']:.2%}**。",
              "", "原图和扫描图是配对变体，不是独立真值。6 份已用于训练的文档及其图片版本仅诊断展示。",
              "", "本结果包含检测、裁切、阅读顺序、视觉转逻辑和识别误差；不修改标注，也不按真值挑选方向。固定 FriBidi base L 与原测试一致，仍可能有句末标点顺序误差。",
              "", "脚本不会额外删除带音标的 03234 文档，该文档仍参与严格主 CER。", "", "[逐图可视化](index.html) · [逐图 CSV](per_image_metrics.csv)"]
    if args.limit_per_group:
        lines.insert(2, f"本次设置了 limit_per_group={args.limit_per_group}，仅为部分图片测试。")
    (args.output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    page = f"<!doctype html><meta charset='utf-8'><title>RTL 测试可视化</title><style>{STYLE}</style><h1>阿拉伯语／希伯来语测试</h1><p>主测试 {len(primary)} 张；用于训练的图片以橙色标识，仅作诊断。图片顶部已显示 CER。</p><p><a href='report.md'>报告</a> · <a href='cer_report.json'>完整 CER</a> · <a href='per_image_metrics.csv'>逐图指标</a></p>"
    page += "<table><tr><th>语言</th><th>图像</th><th>图片数</th><th>CER</th></tr>" + "".join(f"<tr><td>{NAMES[r['language']]}</td><td>{NAMES[r['source']]}</td><td>{r['images']}</td><td>{r['cer']:.2%}</td></tr>" for r in summary) + "</table>"
    for language in LANGUAGES:
        for source in SOURCES:
            page += f"<h2>{NAMES[language]} · {NAMES[source]}</h2><div class='grid'>"
            for row in rows:
                if row["language"] != language or row["source"] != source:
                    continue
                status = "主测试" if row["in_primary_test"] else "训练文档，仅诊断"
                cls = "card" if row["in_primary_test"] else "card excluded"
                page += f"<a class='{cls}' href='{row['group']}/pages/{row['stem']}.html'><img src='{row['group']}/thumbnails/{row['stem']}.jpg'><p>{html.escape(row['image'])}</p><p>{status} · CER {row['cer']:.2%}</p></a>"
            page += "</div>"
    (args.output / "index.html").write_text(page, encoding="utf8")
    return result


def audit(args, inputs, rows):
    if len(inputs) != len(rows):
        raise ValueError("Test coverage mismatch")
    for row in inputs:
        if digest(row["image_path"]) != row["image_sha256"] or digest(row["label_path"]) != row["label_sha256"]:
            raise ValueError(f"Input changed: {row['image']}")
        with Image.open(args.output / row["group"] / "visualized" / (row["stem"] + ".jpg")) as im:
            im.verify()
    for page in args.output.rglob("*.html"):
        for link in re.findall(r"(?:href|src)='([^']+)'", page.read_text(encoding="utf8")):
            if not (page.parent / html.unescape(link)).is_file():
                raise FileNotFoundError(f"Broken HTML link: {page}, {link}")
    write_json(args.output / "audit.json", {"passed": True, "images": len(rows), "input_hashes_unchanged": True,
               "visualizations_decoded": len(rows), "html_links_valid": True, "script_sha256": digest(__file__)})


def main():
    global FONT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paddleocr-root", type=Path, required=True, help="Existing PaddleOCR source root with tools/ and ppocr/")
    parser.add_argument("--rec-model", type=Path, required=True, help="Exported recognition model, not a .pdparams checkpoint")
    parser.add_argument("--det-model", type=Path, required=True, help="Exported official detection model")
    parser.add_argument("--test-data", type=Path, required=True, help="Root containing images_and_label_1019/ and scan_images/")
    parser.add_argument("--dictionary", type=Path, help="Optional dict_no_space.txt; default: exported model's embedded dictionary")
    parser.add_argument("--font", type=Path, help="RTL font; defaults to system DejaVuSans.ttf")
    parser.add_argument("--output", type=Path, default=Path("rtl_test_result"), help="New or empty result directory")
    parser.add_argument("--use-gpu", action="store_true", help="Otherwise use CPU")
    parser.add_argument("--limit-per-group", type=int, default=0, help="Optional smoke-check limit; 0 means all images")
    args = parser.parse_args()
    for key in ["paddleocr_root", "rec_model", "det_model", "test_data", "output"]:
        setattr(args, key, getattr(args, key).resolve())
    if args.limit_per_group < 0:
        parser.error("--limit-per-group must be non-negative")
    if not (args.paddleocr_root / "tools/infer/predict_system.py").is_file() or not (args.paddleocr_root / "ppocr").is_dir():
        parser.error("--paddleocr-root must point to the existing PaddleOCR source checkout")
    if not features.check_feature("raqm"):
        parser.error("Pillow lacks RAQM support required for RTL shaping")
    FONT = choose_font(args.font)
    order = VisualOrder()
    validate_model(args.rec_model)
    validate_model(args.det_model)
    config = yaml.safe_load((args.rec_model / "inference.yml").read_text(encoding="utf8"))
    embedded = config.get("PostProcess", {}).get("character_dict", [])
    chars = args.dictionary.read_text(encoding="utf8").splitlines() if args.dictionary else embedded
    if not chars or any(len(c) != 1 or c == " " for c in chars) or len(set(chars)) != len(chars):
        parser.error("Recognition dictionary must contain unique, single-codepoint entries without a space entry")
    if embedded and chars != embedded:
        parser.error("The supplied dictionary differs from the exported model dictionary")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("--output must be a new or empty folder to avoid mixing models/results")
    inputs = collect_inputs(args)
    args.output.mkdir(parents=True, exist_ok=True)
    dictionary = args.output / "dict_no_space.txt"
    dictionary.write_text("\n".join(chars) + "\n", encoding="utf8")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update({"dictionary_entries_no_space": len(chars), "font_used": FONT, "fribidi": order.version,
                     "rec_image_shape": "3,48,320", "rec_batch_num": 16, "det_limit_side_len": 960, "drop_score": .3,
                     "visual_to_logical_base": "L", "excluded_training_document_stems": sorted(TRAINING_DOCUMENTS),
                     "recognizer_files_sha256": {p.name: digest(p) for p in args.rec_model.iterdir() if p.is_file()}})
    write_json(args.output / "evaluation_config.json", settings)
    write_json(args.output / "inputs.json", inputs)
    cv2.setNumThreads(4)
    system = create_system(args, dictionary)
    rows = []
    with (args.output / "predictions.jsonl").open("w", encoding="utf8") as stream:
        for index, original in enumerate(inputs, 1):
            image = cv2.imread(original["image_path"])
            if image is None:
                raise ValueError(f"Cannot decode: {original['image_path']}")
            boxes, recognition, timings = system(image)
            entries = []
            if boxes is not None:
                for box, (text, confidence) in zip(boxes, recognition):
                    entries.append({"transcription": text, "transcription_visual": text,
                                    "transcription_logical": order(text, "L") if text else "",
                                    "score": float(confidence), "points": np.asarray(box).tolist()})
            prediction = " ".join(e["transcription_logical"].strip() for e in entries if e["transcription_logical"].strip())
            truth = Path(original["label_path"]).read_text(encoding="utf8")
            row = {**original, "ground_truth": normalize(truth), "prediction": normalize(prediction),
                   "boxes": len(entries), **score(prediction, truth)}
            stream.write(json.dumps({**row, "results": entries, "time": timings}, ensure_ascii=False) + "\n")
            stream.flush()
            visualize(args, row, image, entries)
            rows.append(row)
            if index % 10 == 0 or index == len(inputs):
                print(f"Completed {index}/{len(inputs)} images", flush=True)
    result = reports(args, rows)
    audit(args, inputs, rows)
    print(json.dumps({"languages": result["languages"], "overall": result["overall"], "gallery": str(args.output / "index.html")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
