# MZNet 图像去屏纹推理

本项目基于 MZNet 进行改进，用于减少拍摄屏幕时产生的摩尔纹。本目录提供 Python 推理代码和 NCNN 运行库，模型文件另行提供。

推理时将图片切成 512 像素图块，处理后通过重叠融合拼回原图。输出保持原始尺寸，不额外锐化、调色或增强对比度。

## 运行

运行前，请先创建 `models/` 目录，将另行提供的 `model.param` 和 `model.bin` 放入其中，保持文件名不变。本仓库不上传模型文件。

附带的运行库适用于 Linux ARM64，Python 只需安装 NumPy 和 Pillow。在本目录下运行：

```bash
pip install numpy pillow
python infer.py --input /path/input.jpg --output /path/result.png
```

默认最多使用 8 个 CPU 计算线程，更大的图片会自动减少并发，以控制内存占用。结果保存为无损 PNG；为减少保存耗时，默认不压缩，4K 图片约占 25 MB。

## 代码

日常使用只需运行 `infer.py`，其他文件由它自动调用：

- `model.py`：模型加载与推理调度。
- `tiling.py`、`fusion.py`、`fusion_native.py`：分块、融合及加速接口。
- `models/`：运行时放置 NCNN 模型文件，需自行创建，不随仓库上传。
- `native/`：运行库、C++ 源码及 CMake 编译配置。

## 性能与部署

使用 **NCNN 20260113，FP16 权重**。以下结果来自鲲鹏 920 的 8 个物理 CPU 核，输入为原生 4K 图片：

| 项目 | 实测结果 |
| --- | --- |
| NCNN 权重文件 | 约 14.77 MB |
| 完整处理耗时 | 9 次独立进程复测，最慢 9.34 秒 |
| 整程序峰值内存 | 复测最高 663.4 MB（RSS） |

耗时包含读图、模型加载、推理、融合和保存，测试未清空操作系统文件缓存。图片内容、尺寸和设备都会影响速度，以上数字不代表安卓平板性能。

安卓接入时，需要使用项目组的 NCNN 20260113 SDK 和 NDK 重新编译 `native/` 中的代码，并接入分块与融合流程；附带的 Linux 动态库不能直接用于安卓。

模型基于 Moire-Zero / MZNet 官方公开权重改进。NCNN 许可见 `NCNN_LICENSE.txt`。
