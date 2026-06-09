"""
YOLOv8 训练脚本
用法:
    python train.py                          # 默认参数训练
    python train.py --model yolov8s.pt       # 指定模型
    python train.py --epochs 200 --batch 16  # 自定义参数

数据集结构:
    dataset/
    ├── data.yaml          # 数据集配置文件
    ├── train/
    │   ├── images/        # 训练集图片
    │   └── labels/        # 训练集标签 (YOLO格式 .txt)
    └── val/
        ├── images/        # 验证集图片
        └── labels/        # 验证集标签

data.yaml 示例:
    path: ./dataset
    train: train/images
    val: val/images
    names:
      0: cat
      1: dog
"""

import argparse
import os
from ultralytics import YOLO
import torch

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 训练")

    # 模型参数
    parser.add_argument("--model", type=str, default="weights/yolov8s.pt",
                        help="预训练模型路径 (默认: weights/yolov8s.pt)")

    # 数据集参数
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                        help="数据集配置文件路径 (默认: dataset/data.yaml)")

    # 训练参数
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数 (默认: 100)")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小 (默认: 16)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图片尺寸 (默认: 640)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="初始学习率 (默认: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="最终学习率因子 (默认: 0.01, 即 lr0 * lrf)")
    parser.add_argument("--momentum", type=float, default=0.937,
                        help="SGD动量 (默认: 0.937)")
    parser.add_argument("--weight_decay", type=float, default=0.0005,
                        help="权重衰减 (默认: 0.0005)")

    # 优化器与设备
    parser.add_argument("--optimizer", type=str, default="auto",
                        choices=["SGD", "Adam", "AdamW", "auto"],
                        help="优化器 (默认: auto)")
    parser.add_argument("--device", type=str, default="",
                        help="训练设备: 0(GPU0), cpu, 或留空自动选择")

    # 数据增强
    parser.add_argument("--augment", action="store_true", default=False,
                        help="启用更激进的数据增强")

    # 保存与日志
    parser.add_argument("--project", type=str, default="runs/train",
                        help="训练结果保存目录 (默认: runs/train)")
    parser.add_argument("--name", type=str, default="exp",
                        help="实验名称 (默认: exp)")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="从中断处恢复训练")

    # 早停与耐心
    parser.add_argument("--patience", type=int, default=50,
                        help="早停耐心值，多少轮无提升后停止 (默认: 50)")
    parser.add_argument("--workers", type=int, default=8,
                        help="数据加载线程数 (默认: 8)")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 检查数据集配置是否存在 ----
    if not os.path.exists(args.data):
        print(f"[错误] 数据集配置文件不存在: {args.data}")
        print("请确保按以下结构准备数据集:")
        print("  dataset/")
        print("  ├── data.yaml")
        print("  ├── train/images/")
        print("  ├── train/labels/")
        print("  ├── val/images/")
        print("  └── val/labels/")
        return

    print("=" * 60)
    print("YOLOv8 训练配置")
    print("=" * 60)
    print(f"  模型:        {args.model}")
    print(f"  数据集:      {args.data}")
    print(f"  训练轮数:    {args.epochs}")
    print(f"  批次大小:    {args.batch}")
    print(f"  图片尺寸:    {args.imgsz}")
    print(f"  学习率:      {args.lr0}")
    print(f"  设备:        {args.device if args.device else '自动'}")
    print(f"  数据增强:    {'开启' if args.augment else '默认'}")
    print(f"  早停耐心:    {args.patience}")
    print("=" * 60)

    print(f"  是否可以使用显卡推理:    {torch.cuda.is_available()}")  # 应该 True
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))  # 应该显示 RTX 5060

    # ---- 加载模型 ----
    print("\n[1/3] 加载模型...")
    if args.model == "":
        args.model = "weights/yolov8s.pt"
    # 确保模型文件存在
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        print("请使用正确的模型路径，例如: --model weights/yolov8s.pt")
        return
    model = YOLO(args.model)

    # ---- 开始训练 ----
    print("\n[2/3] 开始训练...")
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        device=args.device if args.device else None,
        augment=args.augment,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        # 以下为常用固定配置
        pretrained=True,          # 使用预训练权重
        close_mosaic=10,          # 最后10轮关闭mosaic增强
        cos_lr=True,              # 余弦学习率衰减
        warmup_epochs=3,          # 预热轮数
        exist_ok=True,            # 覆盖同名实验目录
        seed=42,                  # 随机种子
    )

    # ---- 保存最佳模型路径提示 ----
    print("\n[3/3] 训练完成!")
    best_pt = os.path.join(args.project, args.name, "weights", "best.pt")
    print(f"最佳模型: {best_pt}")
    print(f"验证结果:")
    print(f"  mAP50:    {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")


if __name__ == "__main__":
    main()
