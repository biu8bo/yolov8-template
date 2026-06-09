# YOLOv8 命令速查手册

---

## 一、训练 `train.py`

### 常用命令

```bash
# 默认参数训练（yolov8n，100轮）
python train.py

# 指定模型和轮数
python train.py --model weights/yolov8s.pt --epochs 50 --batch 16 --imgsz 1080 --device 0

# 使用 GPU
python train.py --epochs 100 --device 0

# 从中断处恢复训练
python train.py --resume

# 启用额外数据增强，指定实验名称
python train.py --augment --name my_exp
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolov8n.pt` | 预训练模型，可选 yolov8n/s/m/l/x.pt |
| `--data` | `dataset/data.yaml` | 数据集配置文件路径 |
| `--epochs` | `100` | 训练轮数 |
| `--batch` | `16` | 批次大小，显存不足时减小 |
| `--imgsz` | `640` | 输入图片尺寸（必须是 32 的倍数） |
| `--lr0` | `0.01` | 初始学习率 |
| `--lrf` | `0.01` | 最终学习率因子（最终 lr = lr0 × lrf） |
| `--momentum` | `0.937` | SGD 动量 |
| `--weight_decay` | `0.0005` | 权重衰减（L2 正则化） |
| `--optimizer` | `auto` | 优化器，可选 SGD / Adam / AdamW / auto |
| `--device` | 自动 | 训练设备：`0`（GPU0）、`cpu`、留空自动选择 |
| `--augment` | 关闭 | 启用更激进的数据增强（mosaic + mixup） |
| `--project` | `runs/train` | 训练结果保存根目录 |
| `--name` | `exp` | 实验子目录名称 |
| `--resume` | 关闭 | 从上次中断处恢复训练 |
| `--patience` | `50` | 早停耐心值（N 轮无提升则停止） |
| `--workers` | `8` | 数据加载线程数 |

### 输出位置

```
runs/train/exp/
├── weights/
│   ├── best.pt   ← 最佳模型
│   └── last.pt   ← 最后一轮模型
├── results.csv
└── ...
```

---

## 二、推理/预测 `predict.py`

### 常用命令

```bash
# 单张图片推理
python predict.py --model runs/detect/runs/train/exp/weights/best.pt --source test.png

# 使用预训练权重测试
python predict.py --model yolov8n.pt --source test.png

# 批量推理图片目录
python predict.py --model best.pt --source images/

# 推理视频
python predict.py --model best.pt --source video.mp4

# 实时摄像头
python predict.py --model best.pt --source 0

# 调高置信度阈值，保存标签 txt
python predict.py --model best.pt --source test.png --conf 0.5 --save-txt

# 不保存结果，只显示
python predict.py --model best.pt --source test.png --no-save --show
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | （必填）| 模型路径，如 `best.pt` |
| `--source` | （必填）| 输入源：图片/目录/视频/摄像头编号 |
| `--conf` | `0.25` | 置信度阈值，低于此值的框被过滤 |
| `--iou` | `0.7` | NMS IoU 阈值，越小框越少 |
| `--imgsz` | `640` | 推理图片尺寸（须与训练一致） |
| `--device` | 自动 | 推理设备：`0`（GPU）、`cpu`、留空自动 |
| `--save` | 开启 | 保存带标注的推理结果图 |
| `--no-save` | — | 不保存结果（与 `--save` 互斥） |
| `--save-txt` | 关闭 | 同时保存检测框坐标为 `.txt` 文件 |
| `--save-conf` | 关闭 | 在 txt 文件中同时保留置信度 |
| `--save-dir` | `runs/predict` | 结果保存根目录 |
| `--show` | 关闭 | 实时弹窗显示推理画面 |
| `--classes` | 全部 | 只检测指定类别，如 `--classes 0 2` |
| `--line-width` | `2` | 检测框线宽（像素） |
| `--hide-labels` | 关闭 | 隐藏类别标签文字 |
| `--hide-conf` | 关闭 | 隐藏置信度数值 |

---

## 三、模型导出 `export.py`

### 常用命令

```bash
# 默认导出 ONNX（使用训练好的 best.pt）
python export.py

# 指定模型路径
python export.py --model runs/detect/runs/train/exp/weights/best.pt

# 导出 yolov8n.pt 预训练模型
python export.py --model weights/yolov8n.pt

# 动态 batch size（部署时 batch 不固定时使用）
python export.py --dynamic

# FP16 半精度导出（需 GPU，模型体积减半）
python export.py --half --device 0

# 指定 opset 版本（新版 ONNXRuntime 可用 17+）
python export.py --opset 17

# 导出为其他格式
python export.py --format torchscript
python export.py --format openvino
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `runs/detect/runs/train/exp/weights/best.pt` | 待导出的 `.pt` 模型路径 |
| `--format` | `onnx` | 导出格式：`onnx` / `torchscript` / `tflite` / `coreml` / `engine` / `openvino` / `paddle` |
| `--imgsz` | `640` | 导出模型的输入尺寸，须与训练时一致 |
| `--batch` | `1` | 导出时的固定 batch size（`--dynamic` 时忽略） |
| `--device` | `cpu` | 导出设备：`cpu` 或 `0`（GPU） |
| `--half` | 关闭 | 导出 FP16 半精度模型（需 GPU） |
| `--dynamic` | 关闭 | 启用动态 batch/H/W，适合部署时尺寸可变的场景 |
| `--simplify` | 开启 | 精简 ONNX 计算图，去除冗余节点（**推荐保持开启**） |
| `--opset` | `11` | ONNX opset 版本（11 兼容性最广，新版运行时可选 17+） |

### 导出后文件位置

导出文件与源 `.pt` 模型**同目录**，扩展名自动替换：

```
runs/detect/runs/train/exp/weights/best.onnx
```

### ONNXRuntime 快速推理

```bash
pip install onnxruntime      # CPU 推理
# pip install onnxruntime-gpu  # GPU 推理
```

```python
import onnxruntime as ort
import numpy as np
import cv2

session = ort.InferenceSession("best.onnx")
input_name = session.get_inputs()[0].name

img = cv2.imread("test.png")
img = cv2.resize(img, (640, 640))
img = img[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
img = img[np.newaxis]  # -> (1, 3, 640, 640)

outputs = session.run(None, {input_name: img})
# outputs[0] shape: (1, num_anchors, 4+num_classes)
```

---

## 格式对比速查

| 格式 | 参数值 | 适用场景 |
|------|--------|----------|
| ONNX | `onnx` | 通用部署，跨平台，**最常用** |
| TorchScript | `torchscript` | PyTorch C++ 部署 |
| OpenVINO | `openvino` | Intel CPU/VPU 加速 |
| TensorRT | `engine` | NVIDIA GPU 极致加速 |
| TFLite | `tflite` | 移动端 / 嵌入式 |
| CoreML | `coreml` | Apple 设备（macOS/iOS） |
| PaddlePaddle | `paddle` | 百度飞桨生态 |
