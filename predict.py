"""
YOLOv8 预测/推理脚本
用法:
    # 单张图片
    python predict.py --model runs/train/exp/weights/best.pt --source test.jpg

    # 图片目录（批量）
    python predict.py --model best.pt --source images/

    # 视频文件
    python predict.py --model best.pt --source video.mp4

    # 摄像头（实时）
    python predict.py --model best.pt --source 0

    # 指定置信度阈值和保存目录
    python predict.py --model best.pt --source test.jpg --conf 0.5 --save-dir results/
"""

import argparse
import os
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="YOLOv8 预测/推理")

    parser.add_argument("--model", type=str, required=True,
                        help="训练好的模型路径 (如 best.pt)")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源: 图片路径/目录/视频路径/摄像头编号(如0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值 (默认: 0.25)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU阈值 (默认: 0.7)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="推理图片尺寸 (默认: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="推理设备: 0(GPU0), cpu, 或留空自动选择")
    parser.add_argument("--save", action="store_true", default=True,
                        help="保存推理结果 (默认: 开启)")
    parser.add_argument("--no-save", action="store_false", dest="save",
                        help="不保存推理结果")
    parser.add_argument("--save-txt", action="store_true", default=False,
                        help="同时保存检测结果为txt文件")
    parser.add_argument("--save-conf", action="store_true", default=False,
                        help="在txt结果中保留置信度")
    parser.add_argument("--save-dir", type=str, default="runs/predict",
                        help="推理结果保存目录 (默认: runs/predict)")
    parser.add_argument("--show", action="store_true", default=False,
                        help="实时显示推理结果（图片/视频/摄像头时有效）")
    parser.add_argument("--classes", type=int, nargs="+", default=None,
                        help="只检测指定类别 (如: --classes 0 2)")
    parser.add_argument("--line-width", type=int, default=2,
                        help="检测框线宽 (默认: 2)")
    parser.add_argument("--hide-labels", action="store_true", default=False,
                        help="隐藏标签文字")
    parser.add_argument("--hide-conf", action="store_true", default=False,
                        help="隐藏置信度")

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- 判断输入源类型 ----
    source_type = "未知"
    if args.source.isdigit():
        source_type = "摄像头"
    elif os.path.isfile(args.source):
        ext = os.path.splitext(args.source)[1].lower()
        if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
            source_type = "图片"
        elif ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            source_type = "视频"
    elif os.path.isdir(args.source):
        source_type = "图片目录"

    print("=" * 60)
    print("YOLOv8 推理配置")
    print("=" * 60)
    print(f"  模型:        {args.model}")
    print(f"  输入源:      {args.source} ({source_type})")
    print(f"  置信度阈值:  {args.conf}")
    print(f"  IoU阈值:     {args.iou}")
    print(f"  图片尺寸:    {args.imgsz}")
    print(f"  设备:        {args.device if args.device else '自动'}")
    print(f"  结果目录:    {args.save_dir}")
    print("=" * 60)

    # ---- 加载模型 ----
    print("\n加载模型中...")
    model = YOLO(args.model)

    # ---- 执行推理 ----
    print("开始推理...")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device if args.device else None,
        save=args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        project=args.save_dir,
        name="exp",
        exist_ok=True,
        show=args.show,
        classes=args.classes,
        line_width=args.line_width,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
    )

    # ---- 输出结果统计 ----
    total_detections = sum(len(r.boxes) for r in results if r.boxes is not None)
    print(f"\n推理完成! 共处理 {len(results)} 张图，检测到 {total_detections} 个目标")
    print(f"结果已保存至: {args.save_dir}/exp/")


if __name__ == "__main__":
    main()
