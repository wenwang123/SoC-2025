import argparse
from glob import escape
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import torch
ROOT = Path(__file__).resolve().parent

INPUT_PATH = ROOT.parent / "TEST_dewarp"
CHECKPOINT_PATH = ROOT / "checkpoints/checkpoint_best_business_hd95.pth"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_MODE = "production"  # "production" 保存实例 Mask，"debug" 保存叠加图
DEVICE = "cuda"  # "cuda" 使用 BF16，"cpu" 使用 FP32
SCORE_THRESHOLD = 0.4
MAX_DETECTIONS = 8

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLORS = [(90, 80, 240), (80, 210, 70), (245, 160, 35), (220, 70, 180)]


def load_model(checkpoint, device, max_detections):
    from rfdetr import RFDETR
    from rfdetr.datasets.coco import make_coco_transforms_square_div_64

    model = RFDETR.from_checkpoint(checkpoint, device=device, num_select=max_detections)
    config = model.model_config
    if not config.segmentation_head or config.num_classes != 1 or list(model.class_names) != ["document"]:
        raise ValueError("Expected a trained one-class RF-DETR document segmentation checkpoint")
    if not model.model.postprocess.upsample_masks_to_image_size:
        raise ValueError("Checkpoint must restore masks to the original image size")
    model.model.model.to(device).eval()
    transform = make_coco_transforms_square_div_64(
        "val", config.resolution, multi_scale=False,
        patch_size=config.patch_size, num_windows=config.num_windows,
    )
    return model, transform


@torch.inference_mode()
def predict(model, transform, image, threshold):
    width, height = image.size
    device = model.model.device
    tensor, _ = transform(image, None)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
        outputs = model.model.model(tensor.unsqueeze(0).to(device))
        sizes = torch.tensor([[height, width]], device=device)
        result = model.model.postprocess(outputs, sizes)[0]
    keep = (result["labels"] == 0) & (result["scores"].float() >= threshold)
    masks = result["masks"][keep]
    if masks.shape[1:] != (1, height, width) or masks.dtype != torch.bool:
        raise ValueError(f"Expected original-size boolean masks, got {masks.shape}/{masks.dtype}")
    scores = result["scores"][keep].float().cpu().numpy()
    detections = []
    for raw_mask, score in zip(masks[:, 0].cpu().numpy(), scores):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask.astype(np.uint8), connectivity=8)
        if count <= 1:
            continue
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        if stats[component, cv2.CC_STAT_AREA] > 100:
            detections.append((labels == component, float(score)))
    return detections


def save_masks(detections, destination):
    directory = destination.parent
    prefix = f"{destination.name}_instance_"
    directory.mkdir(parents=True, exist_ok=True)
    for index, (mask, _) in enumerate(detections, 1):
        path = directory / f"{prefix}{index:03d}.png"
        temporary = path.with_suffix(".tmp")
        Image.fromarray(mask.astype(np.uint8) * 255).save(temporary, format="PNG")
        temporary.replace(path)
    for old_mask in directory.glob(f"{escape(prefix)}[0-9][0-9][0-9]*.png"):
        number = old_mask.stem.removeprefix(prefix)
        if old_mask.is_file() and number.isdecimal() and int(number) > len(detections):
            old_mask.unlink()


def save_overlay(image, detections, destination):
    overlay = np.asarray(image).copy()
    height = overlay.shape[0]
    for index, (mask, score) in enumerate(detections):
        color = COLORS[index % len(COLORS)]
        overlay[mask] = (overlay[mask] * 0.65 + np.asarray(color) * 0.35).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, max(2, height // 700))
        ys, xs = np.where(mask)
        cv2.putText(overlay, f"doc {index + 1}: {score:.3f}",
                    (int(xs.min()), max(24, int(ys.min()) - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.6, height / 2200), color, 2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(destination, quality=92)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=INPUT_PATH, help="Image or image directory")
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output-mode", choices=("production", "debug"), default=OUTPUT_MODE)
    parser.add_argument("--device", choices=("cuda", "cpu"), default=DEVICE)
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD)
    parser.add_argument("--max-detections", type=int, default=MAX_DETECTIONS)
    args = parser.parse_args(argv)
    if args.output_mode not in ("production", "debug"):
        parser.error('Set OUTPUT_MODE to "production" or "debug" in infer.py')
    if args.device not in ("cuda", "cpu"):
        parser.error('Set DEVICE to "cuda" or "cpu" in infer.py')
    if not 0 <= args.threshold <= 1 or args.max_detections < 1:
        parser.error("threshold must be in [0, 1] and max-detections must be positive")
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    source, output = args.input.resolve(), args.output.resolve()
    if source.is_file() and source.suffix.lower() in IMAGE_SUFFIXES:
        image_root, paths = source.parent, [source]
    elif source.is_dir():
        if output == source or source in output.parents:
            parser.error("Output must be outside the input image directory")
        image_root = source
        paths = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        parser.error(f"Unsupported or missing image input: {source}")
    if not paths:
        parser.error(f"No supported images in {source}")
    if args.device == "cuda" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        parser.error('CUDA mode requires a GPU with BF16 support. Set DEVICE = "cpu" in infer.py for CPU inference.')

    from rfdetr.datasets.io_utils import decode_pil_image

    torch.set_num_threads(4)
    model, transform = load_model(args.checkpoint.resolve(), args.device, args.max_detections)
    for index, path in enumerate(paths, 1):
        image, _ = decode_pil_image(path, draft_size=None)
        detections = predict(model, transform, image, args.threshold)
        key = path.relative_to(image_root)
        if args.output_mode == "production":
            save_masks(detections, output / "production" / key)
        else:
            save_overlay(image, detections, output / "overlays" / key.with_name(key.name + ".jpg"))
        print(f"[{index}/{len(paths)}] {key}: {len(detections)} instances", flush=True)
    print(f"Output: {output}", flush=True)


if __name__ == "__main__":
    main()
