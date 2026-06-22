"""Export CardsOBB model to ONNX with end2end=False (raw 22-channel output).

Usage:
    python export_onnx.py --model best.pt --imgsz 384 640

Output: best.onnx with shape (1, 3, 384, 640) -> (1, 22, N)
"""

import argparse
from ultralytics import CardsYOLO


def main():
    parser = argparse.ArgumentParser(description="Export CardsOBB to ONNX")
    parser.add_argument("model", help="Path to .pt model")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], help="Input size (H W)")
    parser.add_argument("--simplify", action="store_true", default=True, help="Simplify ONNX")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic batch/size")
    parser.add_argument("--half", action="store_true", help="FP16 export")
    args = parser.parse_args()

    model = CardsYOLO(args.model)
    path = model.export(
        format="onnx",
        imgsz=args.imgsz,
        simplify=args.simplify,
        dynamic=args.dynamic,
        half=args.half,
        end2end=False,  # force raw 22-channel output for multi-label postprocessing
    )
    print(f"ONNX exported: {path}")


if __name__ == "__main__":
    main()
