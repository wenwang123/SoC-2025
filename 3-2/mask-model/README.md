# 配对水印 Mask

输入有水印原图和对应的去水印候选图，输出水印概率图和二值 mask。

先安装与设备匹配的 PyTorch 2.7.1、torchvision 0.22.1，再安装其余依赖：

```bash
pip install -r requirements.txt
python verify.py --device cuda
python infer.py --source original.jpg --candidate candidate.png --output outputs --device cuda
```

无显卡时使用 `--device cpu`。默认阈值为 0.35，单方向推理，无额外膨胀或填洞。
两张图应是同一画面，尺寸不同时会自动对齐。

Python 调用：

```python
from PIL import Image
from infer import load_model, predict

model, device, _ = load_model(device="cuda")
result = predict(model, Image.open("original.jpg"), Image.open("candidate.png"), device)
mask = result["probability"] >= 0.35
```

权重位于 `weights/watermark_mask.pt`，校验信息在同名 JSON 文件中。
`visualize.ipynb` 可批量查看四项结果；图对按 `test_data_pairs/<组名>/source` 与
`test_data_pairs/<组名>/candidate` 放置，候选图使用同名 PNG。

模型权重单独交付，不在本仓库中；运行前将其放到 `weights/watermark_mask.pt`。
