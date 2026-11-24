# RF-DETR-Seg 文档实例分割部署

`infer.py` 支持单张照片和批量目录。RF-DETR-Seg Large 根据权重配置将整张图直接缩放到 `1128×1128`，在 NVIDIA GPU 上以 BF16 推理。每个通过筛选的文档对应一个原图尺寸的 mask。

## 在 PyCharm 运行

1. 将项目的 Python 解释器选为已安装 RF-DETR 的环境。本机可以使用现有的 `Instance/RF-DETR-Seg/.venv/bin/python`。
2. 打开 `infer.py`。在文件顶部修改 `OUTPUT_MODE`，按需要设置 `INPUT_PATH`、`CHECKPOINT_PATH`、`OUTPUT_DIR` 和 `DEVICE`。
3. 右键选择 **Run 'infer'**。运行配置中的 **Parameters** 留空即可。

默认读取 `../TEST_dewarp/`，使用部署目录内的 `checkpoints/checkpoint_best_business_hd95.pth`。这份文件来自 RF-DETR 训练第 22 轮 EMA，SHA256 为 `2ea9c3d227fe2194bb533481d749940ccfda872e345d90c717bfae206d702ebe`。网络输入尺寸随权重读取。

| 模式 | 输出 |
| --- | --- |
| `production`（默认） | `outputs/production/<输入子目录>/<原文件名>_instance_001.png` 等。每个实例一张原图尺寸的 0/255 PNG。 |
| `debug` | `outputs/overlays/<原相对路径及文件名>.jpg`。原图尺寸的叠加图包含实例轮廓、序号和置信度。 |

默认按置信度选取最多 8 个候选，保留置信度 ≥0.4 的文档。每个 mask 保留最大 8 连通域，面积须超过 100 个原图像素。对比不同权重时，可以设置不同的 `OUTPUT_DIR`。将 `DEVICE` 改为 `"cpu"` 即可使用 FP32 在 CPU 上运行。

## 新建部署环境

独立部署时使用 Python 3.12，先安装 PyTorch 2.8 / CUDA 12.8 对应的 torch 和 torchvision，再安装 `requirements.txt`，最后用 pip 的 `--no-deps` 选项安装 `requirements-model.txt`，跳过 Supervision 为视频功能声明的 PyAV 依赖。训练工程位于 [Instance/RF-DETR-Seg](../../Instance/RF-DETR-Seg/README.md)。
