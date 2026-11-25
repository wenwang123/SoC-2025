"""Choose upright or 90-degree input using one YOLO pose model at 512 pixels."""
import cv2
import numpy as np
import torch


IMG_SIZE = 512


def _predict(model, image, device):
    result = model.predict(image, imgsz=IMG_SIZE, conf=0.01, device=device, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return None
    index = int(result.boxes.conf.argmax().item())
    box = result.boxes.xyxy[index].cpu().numpy()
    ratio = float((box[2] - box[0]) / max(box[3] - box[1], 1e-6))
    return dict(endpoints=result.keypoints.xy[index].cpu().numpy()[4:6].copy(),
                box_ratio=ratio, confidence=float(result.boxes.conf[index]))


def infer_spine_512(model, image, mode='auto', device=None):
    """Return two spine endpoints in original-image coordinates."""
    if mode not in {'auto', 'upright', 'sideways'}:
        raise ValueError(f'Unsupported mode: {mode}')
    if device is None:
        device = 0 if torch.cuda.is_available() else 'cpu'
    original = _predict(model, image, device) if mode != 'sideways' else None
    rotated = None
    if mode != 'upright':
        rotated_image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        rotated = _predict(model, rotated_image, device)
    if mode == 'upright':
        chosen, rotation = original, 'original'
    elif mode == 'sideways':
        chosen, rotation = rotated, 'cw90'
    else:
        choose_rotated = rotated is not None and (
            original is None or rotated['box_ratio'] > original['box_ratio'] + 0.01 or
            (original['confidence'] < 0.5 and rotated['confidence'] > original['confidence'] + 0.1))
        chosen, rotation = (rotated, 'cw90') if choose_rotated else (original, 'original')
    if chosen is None:
        return None
    endpoints = chosen['endpoints']
    if rotation == 'cw90':
        height = image.shape[0]
        endpoints = np.column_stack([endpoints[:, 1], height - 1 - endpoints[:, 0]])
    return dict(endpoints=endpoints, rotation=rotation,
                confidence=chosen['confidence'])
