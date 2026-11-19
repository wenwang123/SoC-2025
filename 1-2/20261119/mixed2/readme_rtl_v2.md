# PP-OCRv6 RTL 多语言识别 V2

## 训练目标

基于 PaddleOCR `release/3.7` 和官方 `PP-OCRv6_small_rec` 预训练权重，混合微调阿拉伯语（`ar`）、希伯来语（`he`），并增加英语、数字及 RTL 专用标点样本。保持官方 Small 识别架构，没有加入额外 RTL 网络模块。

训练对象为裁切后的单行文字图片。标签采用从左到右的**视觉字符顺序**，图像保持正常方向；整页测试先检测文字区域，再识别，最后将视觉顺序转换为逻辑文本。


## 环境安装

沿用第一版的 `paddleocr37` 环境，不需要为本次路径整理重新安装依赖。

```bash
cd /path/to/PaddleOCR-release37
conda activate paddleocr37

python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda())"
```

本次训练使用 Python 3.9、PaddlePaddle GPU 3.1.1。最小迁移包测试使用 OpenCV、NumPy、Pillow、PyYAML、`rapidfuzz`、系统 FriBidi 和 Pillow 的 RAQM 支持，以及支持阿拉伯语／希伯来语的 DejaVuSans.ttf 字体。

```bash
python -c "import ctypes.util; from PIL import features; print(ctypes.util.find_library('fribidi')); print(features.check_feature('raqm'))"
```


## 数据集构建

以 `mixed2_data_richbg_pdf_visual` 为基础单独建立新版，保留原有合成数据和真实文本行，新增语言、数字及标点区分样本。

```text
data/mixed2_data_20261119/
├── train.txt
├── val.txt
├── dict.txt
├── dict_no_space.txt
├── images/
```

| 项目 | 数量 |
|---|---:|
| 训练样本 | 145,854 |
| 验证样本 | 13,883 |
| `dict_no_space.txt` 条目 | 244 |
| `dict.txt` 条目，含空格 | 245 |
| 新增合成训练／验证样本 | 40,000 ／ 4,000 |
| 新增真实裁切图片 | 150 |

主要调整：

1. 保留旧字符表索引，在末尾追加新字符，补全 Arabic、Hebrew 区块的 19 种专用标点，并使这些标点在训练、验证中都有图像样本。同时补充通用成对标点、希伯来组合符号及波斯形式数字。
2. 按历史 CER 排序并按文档去重，每种语言选取最差的 3 份文档，逐段按画面核对标签。75 个文本片段分别使用原图和扫描图像素生成 150 张训练图片。旧希伯来语页面的错误标注未直接用于这些裁切标签。
3. 新增阿拉伯语单语 6,000 条、希伯来语单语 6,000 条，以及英语、数字、阿英混排、希英混排、RTL 数字、标点、三语混排各 4,000 条；新增验证样本为对应数量的十分之一。
4. 清理训练／验证精确交叉标签、验证重复标签，以及超出 40 个 CTC 时间步的标签。计算时间步时包含连续重复字符需要的 blank；新增合成标签最长为 35 个码点。

训练文件每行格式为：

```text
images/train_word_1.jpg<TAB>视觉顺序标签
images/train_contrast_000001.jpg<TAB>视觉顺序标签
images/train_hard_0001.jpg<TAB>视觉顺序标签
```
训练使用 `dict_no_space.txt` 配合 `use_space_char: true`；最小迁移包默认读取导出模型内置的同一字符表。


## 模型训练

单卡训练命令：

```bash
python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_rtlfix.yml
```

本次最佳模型记录：

```text
best_epoch: 20
acc: 0.958222285559157
norm_edit_dis: 0.9965148914680482
```


## 模型评估

使用最佳权重评估数据集验证集：

```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_rtlfix.yml \
  -o Global.checkpoints=./output/mixed2_small_rec_rtlfix/best_accuracy
```


## 导出推理模型

```bash
python tools/export_model.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_mixed2_rtlfix.yml \
  -o Global.checkpoints=./output/mixed2_small_rec_rtlfix/best_accuracy \
     Global.save_inference_dir=./output/mixed2_small_rec_rtlfix/inference
```

本目录已经包含当前模型的导出结果：

```text
output/mixed2_small_rec_rtlfix/inference/
├── inference.json
├── inference.pdiparams
└── inference.yml
```


## 最小测试迁移包

完整测试只需要执行一个新增脚本。包中只包含新增测试代码和说明。官方 PaddleOCR 代码从目标电脑已有的源码目录加载，不需要将之前的自定义辅助脚本逐个复制或运行。

```text
rtl_test_minimal/
├── run_test.py             # 唯一运行入口
└── README.md               # 最小迁移说明
```

### 迁移需要的文件

| 内容 | 当前项目位置 | 用途 |
|---|---|---|
| 最小脚本包 | `rtl_test_minimal/` | 自定义测试入口 |
| 导出识别模型 | `output/mixed2_small_rec_rtlfix/inference/` | 复制完整目录，包含模型图、参数和内置字典 |
| 导出检测模型 | `inference_models/PP-OCRv6_small_det_infer/` | 官方 Small 检测器 |
| 测试图片及标注 | `ocr_multilangue_testset/` 中以下两个语言目录 | 原图、扫描图及对应真值 |


### 完整运行命令

将脚本包及模型、测试数据放到目标电脑后，激活 PaddleOCR 环境，进入目标电脑的 PaddleOCR 源码根目录。若采用本文目录布局，执行：

```bash
python rtl_test_minimal/run_test.py \
  --paddleocr-root . \
  --rec-model ./output/mixed2_small_rec_rtlfix/inference \
  --det-model ./inference_models/PP-OCRv6_small_det_infer \
  --test-data ./ocr_multilangue_testset \
  --output ./output/mixed2_small_rec_rtlfix_test_minimal \
  --use-gpu
```

所有路径都可以替换为目标电脑的实际位置；脚本包也可以放在源码目录之外。CPU 测试时去掉 `--use-gpu`；若不能自动找到字体，追加 `--font /path/to/DejaVuSans.ttf`。

输出目录必须是新目录或空目录。脚本参数未指定 `--output` 时，默认输出为当前工作目录下的 `rtl_test_result/`。

该命令一次完成：加载已导出模型 → 检测裁切 → 识别 → RTL 视觉序转逻辑序 → CER 计算 → 生成带 CER 标题的图片和 HTML → 检查输入哈希及输出链接。



## 已完成的测试结果

严格 CER 的主测试使用 79 份文档的原图和扫描图，共 158 张图片变体。加入训练的 6 份文档及其两种图片版本不参与主 CER，只在可视化中另作展示。

| 语言 | 图像 | 图片数 | 上一版 CER | 本版 CER |
|---|---|---:|---:|---:|
| 阿拉伯语 | 原图 | 38 | 15.41% | 6.90% |
| 阿拉伯语 | 扫描图 | 38 | 30.23% | 11.21% |
| 希伯来语 | 原图 | 41 | 16.73% | 17.31% |
| 希伯来语 | 扫描图 | 41 | 18.01% | 18.17% |

相同子集总体 micro CER 为 18.45% → 16.02%。这些指标包含检测、裁剪、阅读顺序、RTL 后处理和识别误差，不是纯识别器 CER。

希伯来语 `03234_59cef944a4fd41d1ac01c4729d097c7d_01` 图像有大量音标，标注却未包含音标且存在错字／重复。额外排除该文档的两种图片后，诊断子集结果如下：

| 范围 | 图像 | 图片数 | 上一版 CER | 本版 CER |
|---|---|---:|---:|---:|
| 希伯来语 | 原图 | 40 | 16.47% | 15.86% |
| 希伯来语 | 扫描图 | 40 | 17.77% | 16.75% |
| 阿拉伯语＋希伯来语 | 原图 | 78 | 16.26% | 14.05% |
| 阿拉伯语＋希伯来语 | 扫描图 | 78 | 20.29% | 15.63% |

这是额外排除后的诊断比较，不替代严格主报告。其余 GT、预测和归一化不变，没有通过删除其他预测音标来改变分数。


## RTL 后处理及评测口径

最小迁移包的整页评测固定使用内置 `VisualOrder(text, 'L')`，不根据 GT 为每张图片挑选方向或改写文字。

复现本文件的整页 CER 时，统一使用 `rtl_test_minimal/run_test.py`。原有单行便捷入口采用不同的空格／标点整理逻辑，其结果不能直接混入本口径。

CER = 所有图片 Levenshtein 编辑距离之和 / 所有图片标注字符数之和。计分前双方进行 NFC、移除 `Cf` 格式控制字符并合并空白，保留标点、大小写、数字和组合音标；不是逐图 CER 的算术平均。

字典补全不能自动解决所有标点位置和相似字母错误。当前希伯来语还存在标注质量、句末标点位置、相似字母、小字及真实长行分布差异等问题，后续需按文档隔离训练和测试后再验证改进。
