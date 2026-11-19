# RTL 测试最小迁移包

只有一个需要执行的脚本：`run_test.py`。RTL 转换、CER、可视化、图片顶部 CER 和 HTML 报告已合并，不需要再复制之前的辅助脚本。

本文件夹不包含 PaddleOCR 已有代码、训练代码、模型、训练数据或旧测试结果。

## 另一台电脑需要准备

1. PaddleOCR 源码与其推理环境，需支持 PP-OCRv6 Small；本次使用 PaddlePaddle GPU 3.1.1。脚本调用该源码已有的 `tools.infer.predict_system.TextSystem`。
2. 导出的识别模型：当前项目的 `output/mixed2_small_rec_rtlfix/inference/`，包括 `inference.json`、`inference.pdiparams`、`inference.yml`。不能只复制训练用 `.pdparams`。
3. 导出的检测模型：`inference_models/PP-OCRv6_small_det_infer/`。
4. 测试图片及标注目录 `ocr_multilangue_testset/`，保持下面的结构。
5. `rapidfuzz`、OpenCV、NumPy、PyYAML、Pillow/RAQM、系统 FriBidi，以及支持阿拉伯语／希伯来语的 DejaVuSans.ttf 字体。前几项通常已在 PaddleOCR 环境中；字体可用 `--font` 显式指定。

识别模型的 `inference.yml` 已包含字符表，因此不必迁移整个训练数据集，也不必额外传入字典。如果传入 `--dictionary`，会核对其与模型内置字典完全一致。

```text
ocr_multilangue_testset/
├── images_and_label_1019/
│   ├── 028_heb_希伯来语/     # 原图及同名 .txt 标注
│   └── 029_ara_阿拉伯语/
└── scan_images/
    ├── 028_heb_希伯来语/     # 扫描图，复用原图目录同名 .txt
    └── 029_ara_阿拉伯语/
```

## 只运行一次

将本文件夹复制到另一台电脑，可放在 PaddleOCR 源码之外。激活目标电脑的 PaddleOCR 环境后，按实际位置修改以下四个路径：

```bash
python rtl_test_minimal/run_test.py \
  --paddleocr-root /path/to/PaddleOCR-release37 \
  --rec-model /path/to/mixed2_small_rec_rtlfix/inference \
  --det-model /path/to/PP-OCRv6_small_det_infer \
  --test-data /path/to/ocr_multilangue_testset \
  --output ./rtl_test_result \
  --use-gpu
```

CPU 测试时去掉 `--use-gpu`。若字体不能自动找到，追加 `--font /path/to/DejaVuSans.ttf`。结果目录须为新目录或空目录。

输出：

```text
rtl_test_result/
├── index.html              # 可视化总览
├── report.md               # 分语言、原图／扫描图及总体 CER
├── cer_report.json
├── per_image_metrics.csv
├── predictions.jsonl       # 原始视觉文本、逻辑文本和检测框
├── evaluation_config.json
├── audit.json
└── 各语言分组/
    ├── visualized/         # 左侧输入与检测框，右侧识别文字，顶部 CER
    ├── pages/
    ├── thumbnails/
    └── text/
```

## 与本次测试一致的规则

- 官方 Small 检测器、检测长边 960、识别参考尺寸 3×48×320、动态宽度、batch 16、置信度阈值 0.3，不启用方向分类。
- 视觉转逻辑固定 FriBidi `base L`，不按标注挑选方向；保留当前流程已知的部分句末标点错位限制。
- CER 使用 NFC、移除 Cf、合并空白；保留标点、大小写、数字和音标。总体为编辑距离总和除以标注字符数总和。
- 本模型已训练使用的阿拉伯语 `77214`、`90398`、`90931` 和希伯来语 `37364_…`、`43954_…`、`11721_…` 六份文档会自动从主 CER 排除，但保留诊断可视化。
- 使用当前完整测试集时，主结果为 79 份文档、158 张图片变体；总共生成 170 张可视化。不会额外删除带音标的 `03234_…` 文档。
- 迁移包只测试已导出的模型，不重新训练、导出模型，也不读取旧报告作模型对比。

本包仅包含新增测试代码；官方 PaddleOCR 文件从 `--paddleocr-root` 指定的源码目录加载。
