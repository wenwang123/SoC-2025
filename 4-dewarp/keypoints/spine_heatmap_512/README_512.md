# 中缝检测 512 简化版

这是现有 `spine_heatmap` 推理的并行版本，不覆盖原来的 1024 入口。它去掉单独的方向判断模型，只加载一份 YOLOv8n 和原热力图模型。YOLO 输入为 512，热力图仍处理 384×384 的局部裁图。

默认 `auto` 用同一份 YOLO 权重分别看原图和旋转 90° 的图，再选择更合适的中缝。若调用方已经知道方向，指定 `--mode upright` 或 `--mode sideways`，YOLO 只需运行一次。

## 依赖文件

本次新增 `infer_spine_light_512.py`、`spine_orientation_512.py` 和本说明；复用仓库已有的 `spine_line_heatmap_runtime.py`。权重仍单独交付，解压到本目录后应有：

```text
spine_heatmap_512/weights/detector_512.pt
spine_heatmap_512/weights/heatmap_512.pt
```



## 运行

在 `spine_heatmap_512` 目录执行：

```bash
python infer_spine_light_512.py 输入图片或文件夹 输出文件夹
```

可加 `--mode upright` 或 `--mode sideways`。输出图片中红线是 YOLO 粗中缝，蓝线是热力图细化线；`predictions.csv` 同时保留粗坐标和细化坐标。特殊图若细化失败，可以由调用方使用粗坐标。目前没有自动回退规则。


`config_512.yaml` 是参数记录；实际推理值在 Python 文件中。与原 1024 脚本与配置并行，没有做覆盖，方便比较和回退。
