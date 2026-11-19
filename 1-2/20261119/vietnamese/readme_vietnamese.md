# PP-OCRv6 越南语识别

## 训练目标

基于 PaddleOCR `release/3.7` 源码和官方 `PP-OCRv6_small_rec` 预训练模型微调越南语单行识别模型。重点改善多层附加符号、声调和普通字母的混淆，以及真实打印、扫描图像中的识别效果。保留英文、数字、标点和空格建模。

当前统一使用以下名称，本文所有命令均在仓库根目录执行：

| 项目 | 路径 |
|---|---|
| 数据集 | `data/vietnamese_data_20261118` |
| 配置 | `configs/rec/PP-OCRv6/PP-OCRv6_small_rec_vietnamese.yml` |
| 训练模型 | `output/vietnamese_small_rec_250` |
| 导出推理模型 | `output/vietnamese_small_rec_250/inference` |

## 环境安装

已有 `paddleocr37` 环境时直接激活：

```bash
cd /home/ww/ww/whu/project/PaddleOCR-release37
conda activate paddleocr37
```

新建环境时，可沿用 `readme.md` 中的 Python 3.9、PaddlePaddle GPU 3.1.1 环境：

```bash
conda create -n paddleocr37 python=3.9 -y
conda activate paddleocr37
python -m pip install -U pip setuptools wheel
python -m pip install paddlepaddle-gpu==3.1.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -m pip install -r requirements.txt
```

检查安装结果：

```bash
python -c "import paddle; print(paddle.__version__); print(paddle.device.is_compiled_with_cuda()); print(paddle.device.get_device())"
```

## 数据集构建

训练样本由合成单行图像和指定真实页面的印刷文字裁片组成。合成语料主要使用 Wikipedia 词、短语和短句；自然语料按来源文章划分训练与验证集合。

| 类别 | 训练数量 |
|---|---:|
| 常规合成 | 40,000 |
| 易混字形专项合成 | 25,000 |
| 扫描退化合成 | 25,000 |
| 参考真实测试图样式的合成 | 10,000 |
| 三张指定页面的真实印刷文字裁片 | 121 |
| **训练合计** | **100,121** |
| **独立合成验证集** | **10,000** |

专项样本覆盖 `ế/é/ê`、`ề/è`、`ấ/á/â`、`ố/ó/ô`、`ổ/ỗ`、`ơ/ô`、`ư/u` 以及 `n/r/m/I/l` 等易混字形。图像包含衬线体、粗斜体、深色背景反白文字，以及模糊、压缩和细噪声等打印扫描变化。渲染时为顶部声调和下部圆点保留安全边界。真实裁片仅保留印刷文字，排除手写批注、签名、印章和插图等区域。

数据目录：

```text
data/vietnamese_data_20261118/
├── train.txt
├── val.txt
├── dict.txt
└── images/
    ├── train_word_1.jpg
    ├── ...
    ├── train_word_100121.jpg
    ├── val_word_1.jpg
    ├── ...
    └── val_word_10000.jpg
```

标签格式为 `相对图片路径<TAB>文字`，例如：

```text
images/train_word_1.jpg	PHƯỚC BÌNH
```

图片为 RGB JPEG，高 48 像素、宽度变化。标签使用 NFC Unicode 规范化，长度为 5–30 个字符，含空格、数字和标点。`dict.txt` 有 250 项，配置通过 `use_space_char: true` 另外启用空格。


## 模型训练

官方预训练权重放在：

```text
pretrain_models/PP-OCRv6_small_rec_pretrained.pdparams
```

单卡训练命令：

```bash
python tools/train.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_vietnamese.yml
```

配置从官方预训练权重初始化，`Global.checkpoints` 留空；训练输出到 `output/vietnamese_small_rec_250`。该目录已包含训练完成的模型，再次执行上述命令会向同一目录写入训练结果。

主要训练设置：

| 参数 | 配置 |
|---|---|
| 训练轮数 | 20 |
| 优化器 | Adam |
| 学习率 | 0.0003，Cosine，warmup 2 轮 |
| 多尺度输入 `[W,H]` | `[432,32]`、`[640,48]`、`[856,64]` |
| 验证输入 `[C,H,W]` | `[3,48,640]` |
| CTC 特征时间步 | `rec_train_pool_width: 80` |
| 最大编码长度 | 32，为最长 30 字符文本预留 NRTR BOS/EOS |
| 拼接增强 | 未启用 `RecConAug` |
| 随机上下裁剪 | `RecAug.crop_prob: 0.0`，保护声调与附加符号 |

当前训练日志中的最佳结果：

```text
best_epoch: 20
acc: 0.9981999990018
norm_edit_dis: 0.9997966839201481
```

这里的 `acc` 是合成验证集整行准确率，不能替代真实整页图片的 CER。

## 模型评估

使用最佳权重评估验证集：

```bash
python tools/eval.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_vietnamese.yml \
  -o Global.checkpoints=./output/vietnamese_small_rec_250/best_accuracy
```

`Global.checkpoints` 使用权重前缀，对应文件为 `best_accuracy.pdparams`。

## 导出推理模型

当前目录已包含导出的推理模型；需要重新导出时运行：

```bash
python tools/export_model.py \
  -c configs/rec/PP-OCRv6/PP-OCRv6_small_rec_vietnamese.yml \
  -o Global.checkpoints=./output/vietnamese_small_rec_250/best_accuracy \
     Global.save_inference_dir=./output/vietnamese_small_rec_250/inference
```

导出目录：

```text
output/vietnamese_small_rec_250/inference/
├── inference.json
├── inference.pdiparams
└── inference.yml
```

`inference.yml` 内含导出字典；使用仓库的 `predict_rec.py` / `predict_system.py` 时，仍按下面的命令显式指定数据集字典。当前重命名后的训练权重、导出权重校验值均与本次真实图片评测所用文件一致。

## 推理模型测试

识别数据集中的单行图片：

```bash
python tools/infer/predict_rec.py \
  --use_gpu True \
  --rec_model_dir ./output/vietnamese_small_rec_250/inference \
  --rec_char_dict_path ./data/vietnamese_data_20261118/dict.txt \
  --image_dir ./data/vietnamese_data_20261118/images/val_word_1.jpg
```

测试其他单行图片时替换 `--image_dir`，该参数也支持图片目录。推理保留整行，长行按标准识别预处理动态扩宽，不在空格处分段后拼接。

## 整图测试

完整文档使用官方检测器提取检测区域，再调用本识别模型。以下命令可直接测试越南语扫描图目录：

```bash
python tools/infer/predict_system.py \
  --use_gpu True \
  --image_dir ./ocr_multilangue_testset/scan_images/001_vie_越南语 \
  --det_model_dir ./inference_models/PP-OCRv6_small_det_infer \
  --det_limit_side_len 960 \
  --det_box_type quad \
  --rec_model_dir ./output/vietnamese_small_rec_250/inference \
  --rec_char_dict_path ./data/vietnamese_data_20261118/dict.txt \
  --draw_img_save_dir ./output/vietnamese_small_rec_250/system_test_scan
```

此命令输出检测、识别与标准可视化结果，本身不读取真值或计算 CER；图上的 CER 来自单独的整页评测记录。测试原图时将 `--image_dir` 改为 `./ocr_multilangue_testset/images_and_label_1019/001_vie_越南语`。

## 真实图片 CER

以下为已有评测记录，原图和扫描图各 78 张，对应相同的 78 页内容。识别前只使用官方检测框的标准透视裁片，没有额外分段、扩框或补边。

| 识别模型 | 原图整体 CER | 扫描图整体 CER |
|---|---:|---:|
| 官方 PP-OCRv6 small | 17.1986% | 17.9255% |
| 官方 PP-OCRv5 Latin | 16.9671% | 17.7098% |
| 上次最优 max30/w640 | 13.3166% | 18.3750% |
| **本次越南语模型** | **8.0086%** | **8.3503%** |


本次训练使用了 `12622…`、`72487…`、`09583…` 三页的文字裁片，两组对应页面均视为训练已见页。分开统计如下：

| 图片组 | 未加入训练的 75 页 CER | 训练已见的 3 页 CER |
|---|---:|---:|
| 原图 | 7.6696% | 18.4985% |
| 扫描图 | 8.0373% | 18.0341% |

整体 CER 为所有页面的 Levenshtein 编辑距离之和除以参考字符总数，包含替换、删除和插入；真值与预测均做 NFC、移除格式控制字符、合并连续空白，保留大小写、标点和声调。接受置信度不低于 0.3 的非空识别结果，按检测框排序后以空格连接。整页指标也受到检测遗漏、阅读顺序及标注缺陷影响。

历史标注未修改。例如 `09583…` 原标注存在整行漏标、缺字和邮箱占位问题，因此其历史 CER 会高于实际图文识别错误比例。未加入训练的 75 页也曾用于分析，属于诊断集。
