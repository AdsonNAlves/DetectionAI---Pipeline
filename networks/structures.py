from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, List, Tuple
import random
import os
import glob
import yaml
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from ultralytics import YOLO
import time, csv

# -----------------------------
# YOLO helpers
# -----------------------------
def _default_weights(arch: str) -> str:
    arch = arch.lower().strip()
    if arch in {"v8", "yolov8", "yolo8"}:
        return "yolov8m.pt"
    if arch in {"v11", "yolov11", "yolo11"}:
        return "yolo11m.pt"
    raise ValueError("Arquitetura inválida. Use 'v8' ou 'v11'.")


def _load_model(arch: str, weights: Optional[str | Path] = None) -> YOLO:
    ckpt = str(weights) if weights else _default_weights(arch)
    return YOLO(ckpt)


def _read_yaml(path: Optional[str | Path]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML em {p} deve ser um mapeamento (dict).")
    return data


def train_yolo(
    arch: str,
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> YOLO:
    data_yaml = str(data_yaml)
    train_kwargs = dict(overrides) if overrides is not None else _read_yaml(args_yaml)
    train_kwargs.setdefault("data", data_yaml)
    model = _load_model(arch, weights)
    model.train(**train_kwargs)
    return model


def validate_yolo(
    arch: str,
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
) -> Any:
    data_yaml = str(data_yaml)
    overrides = _read_yaml(args_yaml)
    model = _load_model(arch, weights)
    val_kwargs = dict(overrides)
    val_kwargs.setdefault("data", data_yaml)
    val_kwargs.setdefault("split", "val")
    return model.val(**val_kwargs)


def test_yolo(
    arch: str,
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
) -> Any:
    data_yaml = str(data_yaml)
    overrides = _read_yaml(args_yaml)
    model = _load_model(arch, weights)
    test_kwargs = dict(overrides)
    test_kwargs.setdefault("data", data_yaml)
    test_kwargs.setdefault("split", "test")
    return model.val(**test_kwargs)


# -----------------------------
# Utilidades de caminho/dataset
# -----------------------------
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _list_images(source: str | List[str]) -> List[str]:
    """Aceita um diretório, um arquivo .txt com lista de imagens, um único arquivo, ou uma lista de paths."""
    if isinstance(source, list):
        return source
    p = Path(source)
    if p.is_dir():
        paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(sorted(str(x) for x in p.rglob(ext)))
        return paths
    if p.is_file() and p.suffix.lower() == ".txt":
        with p.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    if p.is_file() and p.suffix.lower() in IMG_EXTS:
        return [str(p)]
    raise FileNotFoundError(f"Fonte de imagens inválida: {source}")


def _infer_labels_path_from_images(images_dir: str | Path) -> str:
    """Ultralytics padrão: .../images/... -> .../labels/..."""
    s = str(images_dir)
    return s.replace("/images", "/labels").replace("\\images", "\\labels")


def _yolo_label_path_for_image(img_path: str, labels_root: Optional[str]) -> str:
    """Obtém o caminho do label .txt com mesmo nome da imagem (YOLO)."""
    img_p = Path(img_path)
    if labels_root:
        # tenta manter estrutura após 'images'
        try:
            parts = list(img_p.parts)
            if "images" in parts:
                idx = parts.index("images")
                sub = Path(*parts[idx + 1:]).with_suffix(".txt")
                return str(Path(labels_root) / sub)
        except Exception:
            pass
        return str(Path(labels_root) / (img_p.stem + ".txt"))
    return str(img_p.with_suffix(".txt")).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")


def _read_yolo_bboxes(label_path: str, img_w: int, img_h: int) -> Tuple[List[List[float]], List[int]]:
    """Lê YOLO txt e converte para Pascal VOC absoluto (xmin,ymin,xmax,ymax). Labels em 1..N."""
    boxes: List[List[float]] = []
    labels: List[int] = []
    if not os.path.exists(label_path):
        return boxes, labels
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            x_c = float(parts[1]) * img_w
            y_c = float(parts[2]) * img_h
            w = float(parts[3]) * img_w
            h = float(parts[4]) * img_h
            xmin = max(0.0, x_c - w / 2.0)
            ymin = max(0.0, y_c - h / 2.0)
            xmax = min(img_w, x_c + w / 2.0)
            ymax = min(img_h, y_c + h / 2.0)
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls + 1)  # torchvision: 1..N (0 é background)
    return boxes, labels


def _resolve_data_yaml_paths(data_yaml: str | Path) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Resolve (images_dir_or_txt, labels_dir) para train/val a partir de data.yaml Ultralytics.
    - Usa d['path'] como base, relativo ao próprio YAML se necessário.
    - Suporta override via env DATA_ROOT.
    """
    d = _read_yaml(data_yaml)
    yaml_path = Path(data_yaml).resolve()

    # base via 'path' do YAML (relativa ao YAML se necessário)
    base_candidate = d.get("path", None)
    if base_candidate:
        base = Path(os.path.expanduser(os.path.expandvars(str(base_candidate))))
        if not base.is_absolute():
            base = (yaml_path.parent / base).resolve()
    else:
        base = yaml_path.parent

    # override por variável de ambiente
    env_root = os.getenv("DATA_ROOT")
    if env_root:
        base = Path(os.path.expanduser(os.path.expandvars(env_root))).resolve()

    def _abs(p: str | None) -> Optional[str]:
        if not p:
            return None
        p_str = os.path.expanduser(os.path.expandvars(str(p)))
        pp = Path(p_str)
        return str(pp if pp.is_absolute() else (base / pp).resolve())

    train_images = _abs(d.get("train"))
    val_images = _abs(d.get("val") or d.get("validation"))
    labels_train = _abs(d.get("labels", None))
    labels_val = _abs(d.get("labels_val", None))

    if train_images and not labels_train:
        labels_train = _infer_labels_path_from_images(train_images)
    if val_images and not labels_val:
        labels_val = _infer_labels_path_from_images(val_images)

    return {
        "train": (train_images, labels_train),
        "val": (val_images, labels_val),
    }


def _device_from_overrides(overrides: Dict[str, Any]) -> torch.device:
    dev = overrides.get("device", None)
    if dev is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(dev, (list, tuple)):
        dev = dev[0]
    if str(dev).lower() == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Dataset (alinhado ao test.ipynb)
# -----------------------------
class SimpleFRCNNDataset(Dataset):
    """
    Dataset simples:
      - Lê imagens e labels YOLO
      - Converte imagem com ToTensor (float32 [0,1], CHW)
      - Mantém boxes em pixels, formato xmin,ymin,xmax,ymax e labels 1..N
    """
    def __init__(
        self,
        images_source: str | List[str],
        labels_root: Optional[str] = None,
        transform: Optional[Any] = None,
    ):
        try:
            self.images: List[str] = _list_images(images_source)
        except Exception:
            self.images: List[str] = _list_images(str(images_source).replace("../../", ""))
        self.labels_root = labels_root
        self.transform = transform if transform is not None else T.ToTensor()

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size

        label_path = _yolo_label_path_for_image(img_path, self.labels_root)
        boxes, labels = _read_yolo_bboxes(label_path, img_w, img_h)

        img_t = self.transform(img_pil)  # tensor float32 [0,1], CHW

        if len(boxes) > 0:
            boxes_t = torch.tensor(boxes, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.int64)
            areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
        else:
            boxes_t = torch.zeros((0, 4), dtype=torch.float32)
            labels_t = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
            "area": areas,
        }
        return img_t, target


def _collate_fn(batch):
    return tuple(zip(*batch))


def _build_frcnn(num_classes: int, pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    # Nota: assinaturas do torchvision podem variar por versão; esta segue a POC
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


# -----
def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AugmentedFRCNNDataset(torch.utils.data.Dataset):
    """
    Wrapper around SimpleFRCNNDataset that applies a random horizontal flip
    (and can be extended) while keeping boxes consistente (expects xyxy tensors).
    Use when you don't have mais complex augmentation logic in your base dataset.
    """
    def __init__(self, base_ds, flip_prob: float = 0.5):
        self.base = base_ds
        self.flip_prob = float(flip_prob)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, target = self.base[idx]  # base returns (PIL/ Tensor image, target dict)
        # ensure img is a Tensor [C,H,W]
        if not torch.is_tensor(img):
            img = T.ToTensor()(img)

        if torch.rand(1).item() < self.flip_prob:
            # horizontal flip on image
            img = torch.flip(img, dims=[2])  # flip width axis (C,H,W)
            # flip boxes: expects target["boxes"] as tensor Nx4 (x1,y1,x2,y2)
            boxes = target.get("boxes", None)
            if boxes is not None and boxes.numel():
                w = float(img.shape[2])
                x1 = w - boxes[:, 2]
                x2 = w - boxes[:, 0]
                boxes = torch.stack([x1, boxes[:, 1], x2, boxes[:, 3]], dim=1)
                target["boxes"] = boxes
        return img, target


# ----

def _box_iou_numpy(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """
    boxes: Nx4 numpy (xmin,ymin,xmax,ymax)
    returns IoU matrix (N1 x N2)
    """
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=float)
    x_min = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y_min = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x_max = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y_max = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter_w = np.clip(x_max - x_min, a_min=0.0, a_max=None)
    inter_h = np.clip(y_max - y_min, a_min=0.0, a_max=None)
    inter = inter_w * inter_h

    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1[:, None] + area2[None, :] - inter
    return inter / (union + 1e-9)


def _compute_ap_from_sorted(tps: np.ndarray, fps: np.ndarray, n_gt: int) -> float:
    """
    Recebe tps/fps arrays já ordenadas por score desc (1D arrays),
    calcula precision-recall e AP via interpolação (area sob curve).
    """
    if n_gt == 0:
        return 0.0
    # cumulative sums
    tp_c = np.cumsum(tps).astype(float)
    fp_c = np.cumsum(fps).astype(float)
    rec = tp_c / (n_gt + 1e-9)
    prec = tp_c / (tp_c + fp_c + 1e-9)

    # add boundary points
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    # make precision monotonic
    for i in range(mpre.size - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    # integrate area
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def compute_detection_metrics(model, val_loader, device, iou_thrs=None, conf_thresh=0.5):
    """
    Returns: precision, recall, map50, map50_95, val_losses_dict
    - model: torchvision FRCNN
    - val_loader: DataLoader yielding (images, targets)
    - device: torch.device
    - iou_thrs: list or numpy array of IoU thresholds e.g. np.arange(0.5,0.96,0.05)
    - conf_thresh: confidence threshold to compute single-point precision/recall
    """

    if iou_thrs is None:
        iou_thrs = np.arange(0.5, 0.96, 0.05)

    # collect per-image GTs and predictions
    all_gt_boxes = []   # list of arrays (Ng x 4)
    all_gt_labels = []  # list of arrays (Ng,)
    all_pred_boxes = [] # list of arrays (Np x 4)
    all_pred_labels = []# list of arrays (Np,)
    all_pred_scores = []# list of arrays (Np,)

    # also accumulate validation losses per component (like your code)
    val_losses = {
        "loss_box_reg": 0.0,
        "loss_classifier": 0.0,
        "loss_objectness": 0.0,
        "loss_rpn_box_reg": 0.0,
    }
    val_steps = 0

    model.eval()
    with torch.no_grad():
        for images, targets in val_loader:
            # images: tuple/list of tensors [C,H,W] (probably on cpu); move to device
            imgs = [img.to(device) for img in images]
            outs = model(imgs)  # list of dicts
            # ensure targets in cpu/numpy
            for t in targets:
                boxes = t.get("boxes", torch.empty((0,4)))
                labels = t.get("labels", torch.empty((0,), dtype=torch.int64))
                all_gt_boxes.append(boxes.cpu().numpy() if isinstance(boxes, torch.Tensor) else np.array(boxes))
                all_gt_labels.append(labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels))

            for o in outs:
                b = o.get("boxes", torch.empty((0,4))).cpu().numpy()
                s = o.get("scores", torch.empty((0,))).cpu().numpy()
                l = o.get("labels", torch.empty((0,), dtype=torch.int64)).cpu().numpy()
                all_pred_boxes.append(b)
                all_pred_labels.append(l)
                all_pred_scores.append(s)

    # compute val losses in train-mode (same as you already do)
    model.train()
    with torch.no_grad():
        for images, targets in val_loader:
            imgs = [img.to(device) for img in images]
            trgs = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(imgs, trgs)
            for k in val_losses.keys():
                val_losses[k] += float(loss_dict.get(k, 0.0))
            val_steps += 1
    # average val losses
    if val_steps > 0:
        for k in val_losses:
            val_losses[k] /= val_steps

    # total number of GTs
    n_gt = sum(len(b) for b in all_gt_boxes)

    # If there are no predictions at all, return zeros
    if len(all_pred_boxes) == 0 or n_gt == 0:
        return 0.0, 0.0, 0.0, 0.0, val_losses

    # Helper: for a single IoU threshold compute AP
    def compute_ap_at_iou(iou_thr):
        scores_list = []
        tps_list = []
        fps_list = []
        for gt_boxes, gt_labels, p_boxes, p_labels, p_scores in zip(
            all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels, all_pred_scores
        ):
            if p_boxes.size == 0:
                continue
            # sort preds by score desc
            order = np.argsort(-p_scores)
            p_boxes_s = p_boxes[order]
            p_labels_s = p_labels[order]
            p_scores_s = p_scores[order]

            matched = np.zeros((len(gt_boxes),), dtype=bool) if len(gt_boxes) > 0 else np.zeros((0,), dtype=bool)
            for pb, pl, ps in zip(p_boxes_s, p_labels_s, p_scores_s):
                scores_list.append(float(ps))
                if gt_boxes.size == 0:
                    tps_list.append(0)
                    fps_list.append(1)
                    continue
                # compute IoU with all gt boxes of same class
                mask = (gt_labels == pl)
                if not mask.any():
                    tps_list.append(0); fps_list.append(1); continue
                candidate_gts = gt_boxes[mask]
                candidate_idx = np.where(mask)[0]
                ious = _box_iou_numpy(pb[None, :], candidate_gts)[0]  # shape (k,)
                best_i = np.argmax(ious)
                best_iou = ious[best_i]
                global_gt_idx = candidate_idx[best_i]
                if best_iou >= iou_thr and not matched[global_gt_idx]:
                    tps_list.append(1); fps_list.append(0); matched[global_gt_idx] = True
                else:
                    tps_list.append(0); fps_list.append(1)
        if len(scores_list) == 0:
            return 0.0
        # sort everything by score desc globally
        order = np.argsort(-np.array(scores_list))
        tps = np.array(tps_list)[order]
        fps = np.array(fps_list)[order]
        ap = _compute_ap_from_sorted(tps, fps, n_gt)
        return ap

    # compute mAP50 and mAP50-95
    map50 = compute_ap_at_iou(0.5)

    # compute APs across iou_thrs
    aps = []
    for thr in iou_thrs:
        aps.append(compute_ap_at_iou(float(thr)))
    map5095 = float(np.mean(aps)) if len(aps) else 0.0

    # precision/recall at conf_thresh (IoU=0.5)
    # recompute TP,FP with only preds having score >= conf_thresh
    tp_total = 0
    fp_total = 0
    total_gt = n_gt
    for gt_boxes, gt_labels, p_boxes, p_labels, p_scores in zip(
        all_gt_boxes, all_gt_labels, all_pred_boxes, all_pred_labels, all_pred_scores
    ):
        # filter preds by conf
        maskp = p_scores >= conf_thresh
        p_boxes_f = p_boxes[maskp]
        p_labels_f = p_labels[maskp]
        p_scores_f = p_scores[maskp]
        if p_boxes_f.size == 0:
            continue
        order = np.argsort(-p_scores_f)
        p_boxes_s = p_boxes_f[order]
        p_labels_s = p_labels_f[order]
        matched = np.zeros((len(gt_boxes),), dtype=bool) if len(gt_boxes) > 0 else np.zeros((0,), dtype=bool)
        for pb, pl in zip(p_boxes_s, p_labels_s):
            if gt_boxes.size == 0:
                fp_total += 1
                continue
            mask = (gt_labels == pl)
            if not mask.any():
                fp_total += 1
                continue
            candidate_gts = gt_boxes[mask]; candidate_idx = np.where(mask)[0]
            ious = _box_iou_numpy(pb[None, :], candidate_gts)[0]
            best_i = np.argmax(ious); best_iou = ious[best_i]; global_gt_idx = candidate_idx[best_i]
            if best_iou >= 0.5 and not matched[global_gt_idx]:
                tp_total += 1
                matched[global_gt_idx] = True
            else:
                fp_total += 1
    fn_total = total_gt - tp_total
    precision = tp_total / (tp_total + fp_total + 1e-9) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total + 1e-9) if (tp_total + fn_total) > 0 else 0.0

    return float(precision), float(recall), float(map50), float(map5095), val_losses


# ----
def train_frcnn(
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Improved Faster R-CNN training tailored for small datasets (~150 imgs).
    - Uses AMP, OneCycleLR by default, gradient clipping, and safe checkpointing.
    - Saves: weights_best.pth (best val_loss) and weights_last.pth
    """
    cfg = dict(overrides) if overrides is not None else _read_yaml(args_yaml)
    # basic hyperparams / defaults tuned for small dataset
    epochs = int(cfg.get("epochs", 50))
    batch = int(cfg.get("batch", 2))
    workers = int(cfg.get("workers", 0))
    lr = float(cfg.get("lr", 1e-3))  # default lower LR
    max_lr = float(cfg.get("max_lr", lr))
    weight_decay = float(cfg.get("weight_decay", 5e-4))
    momentum = float(cfg.get("momentum", 0.9))
    pretrained = bool(cfg.get("pretrained", True))
    seed = int(cfg.get("seed", 42))
    scheduler_type = str(cfg.get("scheduler", "onecycle")).lower()  # 'onecycle'|'multistep'|'cosine'|'steplr'
    grad_clip = float(cfg.get("grad_clip", 5.0))
    flip_prob = float(cfg.get("flip_prob", 0.5))  # simple augmentation
    # score threshold for logging
    score_thresh = float(cfg.get("score_thresh", cfg.get("conf", 0.0) or 0.0))

    set_seed(seed)

    # Resolve dataset info and classes
    dset_info = _read_yaml(data_yaml)
    names = dset_info.get("names", None)
    if "num_classes" in cfg:
        num_classes = int(cfg["num_classes"]) + 1
    elif isinstance(names, (list, tuple)):
        num_classes = len(names) + 1
    elif bool(cfg.get("single_cls", False)):
        num_classes = 2
    else:
        raise ValueError("Defina num_classes em args ou 'names' no data.yaml.")

    device = torch.device("cuda" if torch.cuda.is_available() and str(cfg.get("device", "")).lower() != "cpu" else "cpu")

    # Paths
    paths = _resolve_data_yaml_paths(data_yaml)
    train_images, train_labels = paths["train"]
    val_images, val_labels = paths["val"]

    # Base transform (we keep ToTensor to be safe)
    tensor_tf = T.ToTensor()
    base_train_ds = SimpleFRCNNDataset(train_images, labels_root=train_labels, transform=tensor_tf)
    # wrap with simple augmentation that preserves boxes
    train_ds = AugmentedFRCNNDataset(base_train_ds, flip_prob=flip_prob)

    val_ds = None
    if val_images:
        val_ds = SimpleFRCNNDataset(val_images, labels_root=val_labels, transform=tensor_tf)

    # Dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=_collate_fn) if val_ds else None

    # Model
    model = _build_frcnn(num_classes=num_classes, pretrained=pretrained).to(device)

    # Optimizer: separate weight_decay for biases/BatchNorm
    params_with_decay = []
    params_without_decay = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bias") or p.ndim == 1:
            params_without_decay.append(p)
        else:
            params_with_decay.append(p)
    optimizer = torch.optim.SGD(
        [
            {"params": params_with_decay, "weight_decay": weight_decay},
            {"params": params_without_decay, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=momentum,
    )

    # Scheduler selection
    lr_scheduler = None
    if scheduler_type == "onecycle":
        steps_per_epoch = max(1, len(train_loader))
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=0.3,
            anneal_strategy="cos",
            div_factor=25.0,
            final_div_factor=1e4,
        )
    elif scheduler_type == "multistep":
        milestones = cfg.get("milestones", [int(epochs * 0.6), int(epochs * 0.85)])
        gamma = float(cfg.get("gamma", 0.1))
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    elif scheduler_type == "cosine":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 10), eta_min=lr * 1e-3)
    else:  # steplr fallback
        step_size = int(cfg.get("step_size", max(1, epochs // 3)))
        gamma = float(cfg.get("gamma", 0.1))
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # Save directory
    project = cfg.get("project") or os.getenv("FRCNN_PROJECT") or "checkpoint"
    name = cfg.get("name") or os.getenv("FRCNN_NAME") or "frcnn"
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"[FRCNN] checkpoints em: {save_dir.resolve()}")

    best_val = float("inf")
    best_path = save_dir / "weights_best.pth"
    last_path = save_dir / "weights_last.pth"
    state_path = save_dir / "trainer_state.pth"

    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    try:
        for epoch in range(1, epochs + 1):
            epoch_start = time.time()
            model.train()
            epoch_loss = 0.0
            iters = 0
            for images, targets in train_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type,enabled=(device.type == "cuda")):
                    loss_dict = model(images, targets)
                    loss = sum(loss_dict.values())

                scaler.scale(loss).backward()
                # gradient clipping (unscale first)
                if grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                scaler.step(optimizer)
                scaler.update()

                if isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    lr_scheduler.step()  # OneCycleLR requires step per batch

                epoch_loss += float(loss.item())
                iters += 1

            # step scheduler if not OneCycleLR (per-epoch schedulers)
            if not isinstance(lr_scheduler, torch.optim.lr_scheduler.OneCycleLR):
                lr_scheduler.step()

            avg_train_loss = epoch_loss / max(1, iters)
            msg = f"[FRCNN] Epoch {epoch}/{epochs} - train_loss: {avg_train_loss:.4f}"

            # Validation: 1) log avg preds per img, 2) compute val_loss (using targets) for checkpoint
            val_avg_preds = 0.0
            if val_loader is not None:
                # 1) predictions + metrics + val_losses
                precision, recall, map50, map5095, val_losses = compute_detection_metrics(
                    model, val_loader, device, iou_thrs=np.arange(0.5, 0.96, 0.05), conf_thresh=score_thresh
                )
                # val_losses are dict with keys similar to loss components
                val_box = val_losses.get("loss_box_reg", 0.0)
                val_cls = val_losses.get("loss_classifier", 0.0)
                val_dfl = val_losses.get("loss_objectness", 0.0) + val_losses.get("loss_rpn_box_reg", 0.0)

                epoch_time = round(time.time() - epoch_start, 2)
                msg += f" | val_loss: {val_box+val_cls+val_dfl:.4f}"
                # 2) checkpoint on val combined loss (same logic as before)
                val_loss = val_box + val_cls + val_dfl
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    # atomic-ish save: write to temp and rename
                    tmp_path = best_path.with_suffix(".tmp")
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val": best_val,
                    }, str(tmp_path))
                    tmp_path.replace(best_path)
                    print(f"[FRCNN] novo melhor val_loss={best_val:.4f} -> {best_path.name}")

                # 3) write results.csv in save_dir (YOLO-compatible header)
                results_csv = save_dir / "results.csv"
                header = [
                    "epoch","time","train/box_loss","train/cls_loss","train/dfl_loss",
                    "metrics/precision(B)","metrics/recall(B)","metrics/mAP50(B)","metrics/mAP50-95(B)",
                    "val/box_loss","val/cls_loss","val/dfl_loss","lr/pg0","lr/pg1","lr/pg2"
                ]
                # ensure file exists with header
                if not results_csv.exists():
                    with open(results_csv, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(header)
                # get lr per param group
                lrs = [pg.get("lr", 0.0) for pg in optimizer.param_groups]
                row = [
                    epoch,
                    epoch_time,
                    round(avg_train_loss, 6),
                    0.0,  # placeholder for train/cls_loss averaged (if you tracked separately)
                    0.0,  # placeholder for train/dfl_loss
                    round(precision, 6),
                    round(recall, 6),
                    round(map50, 6),
                    round(map5095, 6),
                    round(val_box, 6),
                    round(val_cls, 6),
                    round(val_dfl, 6),
                    float(lrs[0]) if len(lrs) > 0 else 0.0,
                    float(lrs[1]) if len(lrs) > 1 else 0.0,
                    float(lrs[2]) if len(lrs) > 2 else 0.0,
                ]
                with open(results_csv, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(row)

                # append info to msg for console
                msg += f" | prec:{precision:.3f} rec:{recall:.3f} mAP50:{map50:.3f} mAP50-95:{map5095:.3f}"

            print(msg)

        # final save
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict(),
            "best_val": best_val,
        }, str(last_path))
        # also save trainer state for resuming
        torch.save({
            "epoch": epochs,
            "best_val": best_val,
        }, str(state_path))

        return model

    except KeyboardInterrupt:
        # safe save on interrupt: salva checkpoint completo em weights_last.pth e trainer_state.pth
        print("[FRCNN] interrupção por KeyboardInterrupt — salvando estado seguro...")
        # pega epoch atual de forma segura (0 se não definido)
        curr_epoch = locals().get("epoch", 0)

        # salva checkpoint completo (model + optimizer + scheduler + scaler + best_val) atomicamente
        tmp_last = last_path.with_suffix(".tmp")
        torch.save({
            "epoch": curr_epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if "optimizer" in locals() else None,
            "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if "scaler" in locals() else None,
            "best_val": best_val,
        }, str(tmp_last))
        tmp_last.replace(last_path)

        # salva trainer_state (para retomar) também atomicamente
        tmp_state = state_path.with_suffix(".tmp")
        torch.save({
            "epoch": curr_epoch,
            "best_val": best_val,
        }, str(tmp_state))
        tmp_state.replace(state_path)

        raise

def validate_frcnn(
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Validação simples: roda inferência no split=val e retorna predições (no CPU).
    """
    cfg = dict(overrides) if overrides is not None else _read_yaml(args_yaml)
    batch = int(cfg.get("batch", 4))
    workers = int(cfg.get("workers", 0))
    device = _device_from_overrides(cfg)

    paths = _resolve_data_yaml_paths(data_yaml)
    val_images, val_labels = paths["val"]
    if not val_images:
        raise ValueError("data_yaml não contém 'val'.")

    tensor_tf = T.ToTensor()
    val_ds = SimpleFRCNNDataset(val_images, labels_root=val_labels, transform=tensor_tf)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=_collate_fn)

    dset_info = _read_yaml(data_yaml)
    names = dset_info.get("names", None)
    num_classes = len(names) + 1 if isinstance(names, (list, tuple)) else 2

    model = _build_frcnn(num_classes=num_classes, pretrained=True)
    if weights and os.path.exists(str(weights)):
        model.load_state_dict(torch.load(str(weights), map_location="cpu"))
    model.to(device).eval()

    all_outputs: List[Any] = []
    with torch.no_grad():
        for images, _ in val_loader:
            images = [img.to(device) for img in images]
            outs = model(images)
            all_outputs.extend([{k: v.cpu() for k, v in o.items()} for o in outs])
    return all_outputs


# -----------------------------
# Dispatcher
# -----------------------------
def train_model(
    arch: str,
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    arch_l = arch.lower().strip()
    if arch_l in {"v8", "yolov8", "yolo8", "v11", "yolov11", "yolo11"}:
        return train_yolo(arch_l.replace("yolo", "").replace("yolov", "v"), data_yaml, args_yaml, weights, overrides)
    if arch_l in {"frcnn", "faster", "fasterrcnn"}:
        return train_frcnn(data_yaml, args_yaml, overrides)
    raise ValueError("Arquitetura inválida. Use 'v8', 'v11' ou 'frcnn'.")


__all__ = [
    "train_yolo",
    "validate_yolo",
    "test_yolo",
    "train_frcnn",
    "validate_frcnn",
    "train_model",
]
