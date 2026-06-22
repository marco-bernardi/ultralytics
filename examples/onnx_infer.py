"""Standalone ONNX inference for CardsOBB multi-label model (no ultralytics dependency).

Pipeline:
1. export_onnx.py  -> exports best.pt to best.onnx (end2end=False, raw 22-channel output)
2. onnx_infer.py   -> runs onnxruntime + standalone postprocessing (suit/rank split + rotated NMS)

ONNX output shape: (1, 22, N) = [box_xywh(4), cls_sigmoid(17), angle(1)] in BCN format.
"""

import argparse
import cv2
import numpy as np
import onnxruntime as ort


NAMES = ["S", "D", "C", "H", "A", "2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K"]
NC_SUIT = 4  # channels 0-3
NC_RANK = 13  # channels 4-16
NC_TOTAL = 17


def letterbox(img, new_shape=(384, 640), color=(114, 114, 114)):
    """Resize image to fit new_shape with letterbox padding. Returns (img, ratio, pad)."""
    h, w = img.shape[:2]
    nh, nw = new_shape
    r = min(nh / h, nw / w)
    nh2, nw2 = int(round(h * r)), int(round(w * r))
    img = cv2.resize(img, (nw2, nh2), interpolation=cv2.INTER_LINEAR)
    top = (nh - nh2) // 2
    bottom = nh - nh2 - top
    left = (nw - nw2) // 2
    right = nw - nw2 - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, (r, r), (left, top)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def batch_probiou(obb1, obb2, eps=1e-7):
    """Calculate IoU between rotated boxes (xywhr) using probabilistic IoU.

    Args:
        obb1: (N, 5) xywhr
        obb2: (M, 5) xywhr
    Returns:
        (N, M) IoU matrix
    """
    x1, y1, w1, h1, t1 = np.split(obb1, 5, axis=-1)
    x2, y2, w2, h2, t2 = np.split(obb2, 5, axis=-1)
    t1 = t1 % np.pi
    t2 = t2 % np.pi
    # Convert to covariance
    a1 = (w1 / 2) ** 2 + eps
    b1 = (h1 / 2) ** 2 + eps
    c1 = np.cos(t1)
    s1 = np.sin(t1)
    a2 = (w2 / 2) ** 2 + eps
    b2 = (h2 / 2) ** 2 + eps
    c2 = np.cos(t2)
    s2 = np.sin(t2)
    # Bhattacharyya distance
    dx = x1 - x2.T
    dy = y1 - y2.T
    t1_cos2 = c1 ** 2
    t1_sin2 = s1 ** 2
    t2_cos2 = c2.T ** 2
    t2_sin2 = s2.T ** 2
    # Simplified probiou (Gaussian Wasserstein distance approximation)
    w_dist = (t1_cos2 * a1 + t1_sin2 * b1 + t2_cos2 * a2 + t2_sin2 * b2) * eps
    h_dist = (t1_sin2 * a1 + t1_cos2 * b1 + t2_sin2 * a2 + t2_cos2 * b2) * eps
    # Diagonal elements
    d1 = (a1 * t1_cos2 + b1 * t1_sin2 + a2.T * t2_cos2 + b2.T * t2_sin2) * eps
    d2 = (a1 * t1_sin2 + b1 * t1_cos2 + a2.T * t2_sin2 + b2.T * t2_cos2) * eps
    # Off-diagonal
    cross = ((a1 - b1) * c1 * s1 + (a2 - b2).T * c2 * s2) * eps
    # Wasserstein distance
    bd2 = (dx ** 2) / (d1 + eps) + (dy ** 2) / (d2 + eps)
    hd = (w_dist + h_dist + 2 * np.abs(cross)) / (np.sqrt(d1 * d2) + eps)
    bd2 = bd2 + 0.5 * hd
    iou = np.exp(-bd2.clip(min=0))
    return iou.squeeze() if iou.size > 1 else iou


def rotated_nms(boxes_xywhr, scores, iou_thres=0.45):
    """Simple rotated NMS using probiou. Returns kept indices."""
    if len(scores) == 0:
        return np.array([], dtype=int)
    order = scores.argsort()[::-1]
    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        iou = batch_probiou(boxes_xywhr[i:i+1], boxes_xywhr[rest])
        iou = np.asarray(iou).flatten()
        order = rest[iou < iou_thres]
    return np.array(keep, dtype=int)


def postprocess(preds, conf_thres=0.25, iou_thres=0.45, max_det=300, max_wh=7680):
    """Postprocess CardsOBB ONNX output (1, 22, N) -> list of detections.

    Each detection: [x, y, w, h, conf, cls, angle] where cls is 0-16 (suit or rank).
    Each card produces 2 detections (one suit, one rank) with identical xywhr.
    """
    # preds: (1, 22, N) BCN -> transpose to (1, N, 22) BNC
    pred = preds[0].T  # (N, 22)
    box = pred[:, :4]  # xywh
    cls = sigmoid(pred[:, 4:21])  # 17 sigmoid scores
    angle = pred[:, 21:22]  # angle

    # Best suit and rank per anchor
    suit_conf = cls[:, 0:4].max(axis=1)
    suit_id = cls[:, 0:4].argmax(axis=1)
    rank_conf = cls[:, 4:17].max(axis=1)
    rank_id = cls[:, 4:17].argmax(axis=1) + 4  # local 0-12 -> global 4-16

    # Independent filtering: keep anchor if either suit or rank exceeds threshold
    s_keep = suit_conf > conf_thres
    r_keep = rank_conf > conf_thres

    detections = []

    # Suit detections
    if s_keep.any():
        box_s = box[s_keep]
        angle_s = angle[s_keep]
        conf_s = suit_conf[s_keep]
        cls_s = suit_id[s_keep]
        detections.append((box_s, angle_s, conf_s, cls_s))

    # Rank detections
    if r_keep.any():
        box_r = box[r_keep]
        angle_r = angle[r_keep]
        conf_r = rank_conf[r_keep]
        cls_r = rank_id[r_keep]
        detections.append((box_r, angle_r, conf_r, cls_r))

    if not detections:
        return np.zeros((0, 7))

    boxes2 = np.concatenate([d[0] for d in detections], axis=0)
    angle2 = np.concatenate([d[1] for d in detections], axis=0)
    conf2 = np.concatenate([d[2] for d in detections], axis=0)
    cls2 = np.concatenate([d[3] for d in detections], axis=0).astype(float)

    if len(boxes2) == 0:
        return np.zeros((0, 7))

    # Rotated NMS per class: offset xy by cls*max_wh
    c = cls2 * max_wh
    boxes_nms = np.concatenate([boxes2[:, :2] + c[:, None], boxes2[:, 2:4], angle2], axis=-1)
    idx = rotated_nms(boxes_nms, conf2, iou_thres)[:max_det]

    det = np.concatenate([boxes2[idx], conf2[idx, None], cls2[idx, None], angle2[idx]], axis=-1)
    return det  # (n, 7): x, y, w, h, conf, cls, angle


def scale_boxes(boxes, img_shape, orig_shape, ratio_pad):
    """Scale boxes from model input to original image."""
    r, (pad_w, pad_h) = ratio_pad
    boxes[:, 0] = (boxes[:, 0] - pad_w) / r
    boxes[:, 1] = (boxes[:, 1] - pad_h) / r
    boxes[:, 2] = boxes[:, 2] / r
    boxes[:, 3] = boxes[:, 3] / r
    return boxes


def xywhr_to_xyxyxyxy(xywhr):
    """Convert (x, y, w, h, angle) to 4-corner polygon (8,)."""
    x, y, w, h, a = xywhr
    cos_a, sin_a = np.cos(a), np.sin(a)
    dx, dy = w / 2, h / 2
    corners = np.array([
        [-dx, -dy], [dx, -dy], [dx, dy], [-dx, dy]
    ])
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners @ R.T
    corners[:, 0] += x
    corners[:, 1] += y
    return corners.flatten()


def infer(onnx_path, img_path, conf=0.25, iou=0.45, imgsz=(384, 640)):
    """Run ONNX inference on a single image. Returns list of (suit_name, rank_name, conf, polygon)."""
    # Load model
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    # Preprocess
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    img_lb, ratio, pad = letterbox(img, new_shape=imgsz)
    img_lb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_lb = img_lb.transpose(2, 0, 1).astype(np.float32) / 255.0
    img_lb = img_lb[np.newaxis]  # (1, 3, H, W)

    # Inference
    preds = session.run(None, {input_name: img_lb})[0]  # (1, 22, N)

    # Postprocess
    det = postprocess(preds, conf_thres=conf, iou_thres=iou)
    if det.shape[0] == 0:
        return []

    # Scale to original image
    det[:, :4] = scale_boxes(det[:, :4], imgsz, (orig_h, orig_w), (ratio, pad))

    # Group suit+rank by identical xywhr (same card)
    cards = {}
    for d in det:
        x, y, w, h, c, cls_id, a = d
        key = (round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(a, 3))
        cid = int(cls_id)
        poly = xywhr_to_xyxyxyxy([x, y, w, h, a])
        if key not in cards:
            cards[key] = {"suit": None, "rank": None, "conf": float(c), "poly": poly}
        if cid < NC_SUIT:
            cards[key]["suit"] = NAMES[cid]
        else:
            cards[key]["rank"] = NAMES[cid]

    results = []
    for key, info in cards.items():
        results.append((info["suit"], info["rank"], info["conf"], info["poly"]))
    return results


def main():
    parser = argparse.ArgumentParser(description="CardsOBB ONNX inference (standalone)")
    parser.add_argument("model", help="Path to .onnx model")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--imgsz", type=int, nargs=2, default=[384, 640], help="Input size (H W)")
    args = parser.parse_args()

    results = infer(args.model, args.image, conf=args.conf, iou=args.iou, imgsz=tuple(args.imgsz))

    if not results:
        print("Nessuna detection")
        return

    print(f"{len(results)} carte rilevate:")
    for suit, rank, conf, poly in results:
        print(f"  {suit}{rank} conf={conf:.3f} poly={poly.tolist()}")


if __name__ == "__main__":
    main()
