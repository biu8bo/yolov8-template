"""
YOLOv8 模型导出脚本
支持导出为 ONNX、TensorRT、CoreML 等格式

用法:
    python export.py                                        # 默认导出 ONNX
    python export.py --model runs/detect/.../best.pt       # 指定模型
    python export.py --format onnx --imgsz 640             # 指定格式和尺寸
    python export.py --format onnx --dynamic               # 动态 batch size
    python export.py --format onnx --simplify              # 简化计算图（推荐）
"""

import argparse
import os
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 模型导出")

    parser.add_argument(
        "--model",
        type=str,
        default="runs/detect/runs/train/exp/weights/best.pt",
        help="待导出的 .pt 模型路径 (默认: runs/detect/runs/train/exp/weights/best.pt)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        choices=["onnx", "torchscript", "tflite", "coreml", "engine", "openvino", "paddle"],
        help="导出格式 (默认: onnx)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸 (默认: 640)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="导出时的 batch size (默认: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="导出设备: cpu 或 0(GPU) (默认: cpu)",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="导出 FP16 半精度模型（需要 GPU）",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="ONNX: 启用动态 batch/height/width",
    )
    parser.add_argument(
        "--simplify",
        action="store_true",
        default=True,
        help="ONNX: 简化计算图，减少冗余节点 (默认: 开启)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=11,
        help="ONNX opset 版本 (默认: 11，兼容性最广)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 检查模型文件是否存在 ----
    if not os.path.exists(args.model):
        print(f"[错误] 模型文件不存在: {args.model}")
        print("请先训练模型，或手动指定 --model 路径")
        return

    print("=" * 60)
    print("YOLOv8 模型导出配置")
    print("=" * 60)
    print(f"  模型路径:    {args.model}")
    print(f"  导出格式:    {args.format.upper()}")
    print(f"  输入尺寸:    {args.imgsz}x{args.imgsz}")
    print(f"  Batch Size:  {args.batch}")
    print(f"  设备:        {args.device}")
    print(f"  半精度(FP16):{args.half}")
    if args.format == "onnx":
        print(f"  动态尺寸:    {args.dynamic}")
        print(f"  简化图:      {args.simplify}")
        print(f"  Opset 版本:  {args.opset}")
    print("=" * 60)

    # ---- 加载模型 ----
    print("\n[1/2] 加载模型...")
    model = YOLO(args.model)

    # ---- 执行导出 ----
    print(f"\n[2/2] 导出为 {args.format.upper()} 格式...")
    export_path = model.export(
        format=args.format,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        half=args.half,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset=args.opset,
    )

    print("\n" + "=" * 60)
    print("导出完成!")
    print(f"导出路径: {export_path}")
    print("=" * 60)

    # ---- ONNX 额外信息提示 ----
    if args.format == "onnx":
        print("\n[提示] ONNX 使用方式:")
        print("  pip install onnxruntime  # CPU 推理")
        print("  pip install onnxruntime-gpu  # GPU 推理")
        print()
        print("  import onnxruntime as ort")
        print("  import numpy as np")
        print(f'  session = ort.InferenceSession("{export_path}")')
        print(f'  input_name = session.get_inputs()[0].name')
        print(f'  # 输入 shape: (1, 3, {args.imgsz}, {args.imgsz}), float32, 值域 [0,1]')
        print(f'  outputs = session.run(None, {{input_name: img_array}})')


if __name__ == "__main__":
    main()
