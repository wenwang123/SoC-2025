# 图像模糊去除

## 文档
- [算法设计文档](doc/图像模糊去除算法设计文档.docx)
- [自测试报告](doc/图像模糊去除自测试报告.docx)

## 版本基线

- 训练与Paddle推理基线：`code/test/model.pdparams`，NAFNet `width=24`，参数量约4.92M。
- 0416 NCNN部署产物：首层卷积为32通道，参数量约8.70M，与上述Paddle权重不是同一版本。
- 固定复测使用最后100对验证图像，按0.5倍缩放后取中心1024×1024区域。详细结果见自测试报告和`test-result`目录。

## 环境配置
```sh
pip install -r doc\requirements.txt
```

## 数据预处理
```shell
python -m code.data.data_preparation
python -m code.data.data_preparation_all

python -m code.data.generate_meta_info
python -m code.data.generate_meta_info_all
```

## 训练第一阶段
```sh
python code\train.py -opt code\configs\NAFNet_phase1.yml
```

## 训练第二阶段
```sh
python code\train.py -opt code\configs\NAFNet_phase2.yml --resume output\model\last_model
```

## 全量数据微调
```sh
python code\train.py -opt code\configs\NAFNet_fintune_all.yml --resume output\model\best_model
```

## 测试脚本
```sh
python code\test\predict.py Data\Val_data myResults
```
