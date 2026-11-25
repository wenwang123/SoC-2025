"""Single YOLOv8n (512) plus the existing local spine heatmap.

Run: python infer_spine_light_512.py INPUT OUTPUT
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from spine_orientation_512 import infer_spine_512
from spine_line_heatmap_runtime import SpineLineNet, refine_endpoints


ROOT = Path(__file__).resolve().parent
DETECTOR = ROOT / 'weights/detector.pt' if (ROOT / 'weights/detector.pt').is_file() else ROOT / 'runs_6kpts/pose_1024_orientation_normalized_20261102/weights/best.pt'
HEATMAP = ROOT / 'weights/heatmap.pt' if (ROOT / 'weights/heatmap.pt').is_file() else ROOT / 'runs_6kpts/spine_line_heatmap_20261103/best.pt'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path, help='Image or folder containing JPG/PNG images')
    parser.add_argument('output', type=Path)
    parser.add_argument('--detector', type=Path, default=DETECTOR)
    parser.add_argument('--heatmap', type=Path, default=HEATMAP)
    parser.add_argument('--mode', choices=('auto', 'upright', 'sideways'), default='auto',
                        help='auto tests original and 90-degree input with one YOLO model')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = 0 if torch.cuda.is_available() else 'cpu'
    heatmap_device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    detector = YOLO(str(args.detector))
    heatmap = SpineLineNet(pretrained=False)
    checkpoint = torch.load(args.heatmap, map_location='cpu', weights_only=True)
    heatmap.load_state_dict(checkpoint['model'])
    heatmap.eval().to(heatmap_device)
    images = ([args.input] if args.input.is_file() else sorted(
        p for p in args.input.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}))
    rows = []
    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f'Cannot read image: {path}')
        name = path.name if args.input.is_file() else path.relative_to(args.input).as_posix()
        prediction = infer_spine_512(detector, image, mode=args.mode, device=device)
        preview = image.copy()
        if prediction is None:
            cv2.putText(preview, 'NO DETECTION', (30, 70), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3)
            rows.append([name, 'miss', '', *([''] * 8)])
        else:
            coarse = prediction['endpoints']
            refined, meta = refine_endpoints(heatmap, image, coarse, device=heatmap_device)
            for points, color in ((coarse, (0, 0, 255)), (refined, (255, 0, 0))):
                points = np.rint(points).astype(int)
                cv2.line(preview, tuple(points[0]), tuple(points[1]), color,
                         max(3, round(min(image.shape[:2]) / 300)))
            rows.append([name, prediction['rotation'], meta['line_confidence'],
                         *[float(x) for x in coarse.flatten()],
                         *[float(x) for x in refined.flatten()]])
        destination = args.output / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(destination), preview):
            raise RuntimeError(f'Cannot save preview: {destination}')
    with (args.output / 'predictions.csv').open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'rotation', 'line_confidence', 'coarse_x1', 'coarse_y1',
                         'coarse_x2', 'coarse_y2', 'refined_x1', 'refined_y1',
                         'refined_x2', 'refined_y2'])
        writer.writerows(rows)
    print(f'Processed {len(rows)} images; input 512, single YOLO + heatmap; output: {args.output}')


if __name__ == '__main__':
    main()
