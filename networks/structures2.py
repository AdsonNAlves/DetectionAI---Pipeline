from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml
from ultralytics import YOLO

from common.utils import (
    get_train_transforms,
    get_val_transforms,
    apply_transforms,
    prepare_frcnn_sample,
)

# -----------------------------
# Dataset para Faster R-CNN
# -----------------------------

# -----------------------------
# Funções auxiliares YOLO (mesmo padrão de structures.py)
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
	"""Lê um YAML e retorna dict; se não existir ou for None, retorna {}."""
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

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

def _list_images(source: str | List[str]) -> List[str]:
    """Aceita um diretório, um arquivo .txt com lista de imagens, ou uma lista de paths."""
    if isinstance(source, list):
        return source
    p = Path(source)
    if p.is_dir():
        paths: List[str] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            paths.extend(sorted(str(x) for x in p.rglob(ext)))
        return paths
    if p.is_file() and p.suffix.lower() in {".txt"}:
        with p.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    # se for um único arquivo de imagem
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
        # tenta reconstruir estrutura padrão .../images/sub/... -> .../labels/sub/...
        try:
            parts = list(img_p.parts)
            # encontra o índice de 'images' para manter subpastas após
            if "images" in parts:
                idx = parts.index("images")
                sub = Path(*parts[idx+1:]).with_suffix(".txt")
                return str(Path(labels_root) / sub)
        except Exception:
            pass
        return str(Path(labels_root) / (img_p.stem + ".txt"))
    else:
        # substitui 'images' por 'labels' no caminho completo
        return str(img_p.with_suffix(".txt")).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\")


def _read_yolo_bboxes(label_path: str, img_w: int, img_h: int) -> Tuple[List[List[float]], List[int]]:
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
                # torchvision espera labels 1..N (0 é background)
                labels.append(cls + 1)
    return boxes, labels


class AlbumentationsFRCNNDataset(Dataset):
    """
    Dataset para Faster R-CNN lendo imagens e rótulos no formato YOLO (Ultralytics).
    Aplica Albumentations (get_train_transforms/get_val_transforms) via apply_transforms antes de entregar ao modelo.
    """
    def __init__(
        self,
        images_source: str | List[str],
        labels_root: Optional[str] = None,
        transforms: Optional[Any] = None,
    ):
        try:
            self.images: List[str] = _list_images(images_source)
        except:
            self.images: List[str] = _list_images(images_source.replace("../../",""))
        self.labels_root = labels_root
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img_pil = Image.open(img_path).convert("RGB")
        img_w, img_h = img_pil.size

        label_path = _yolo_label_path_for_image(img_path, self.labels_root)
        boxes, labels = _read_yolo_bboxes(label_path, img_w, img_h)

        # Albumentations trabalha com numpy (HWC, RGB)
        img_np = np.array(img_pil)

        # Delega o processamento para utils.prepare_frcnn_sample
        img_tensor, boxes_t, labels_t = prepare_frcnn_sample(img_np, boxes, labels, self.transforms)

        target = {
            "boxes": boxes_t,
            "labels": labels_t,
            "image_id": torch.tensor([idx]),
            "iscrowd": torch.zeros((boxes_t.shape[0],), dtype=torch.int64),
        }
        if boxes_t.numel():
            areas = (boxes_t[:, 2] - boxes_t[:, 0]) * (boxes_t[:, 3] - boxes_t[:, 1])
            target["area"] = areas
        else:
            target["area"] = torch.zeros((0,), dtype=torch.float32)

        return img_tensor, target


def _collate_fn(batch):
    return tuple(zip(*batch))


def _build_frcnn(num_classes: int, pretrained: bool = True) -> torchvision.models.detection.FasterRCNN:
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def _resolve_data_yaml_paths(data_yaml: str | Path) -> Dict[str, Tuple[str, Optional[str]]]:
    """
    Lê um data.yaml (Ultralytics) e resolve (images_dir_or_txt, labels_dir) para train/val.
    Se labels_dir não for definido, será inferido trocando 'images' -> 'labels'.
    """
    d = _read_yaml(data_yaml)
    base = Path(d.get("path", "."))
    def _abs(p: str | None) -> Optional[str]:
        if not p:
            return None
        pp = Path(p)
        return str(pp if pp.is_absolute() else base / pp)

    train_images = _abs(d.get("train"))
    val_images = _abs(d.get("val") or d.get("validation"))
    labels_train = _abs(d.get("labels", None))  # opcional (não padrão), caso exista
    labels_val = _abs(d.get("labels_val", None))  # opcional

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
    if str(dev).lower() in {"cpu"}:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_frcnn(
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Treina Faster R-CNN com pré-processamento de get_train_transforms (Albumentations).
    - data_yaml: data.yaml no formato Ultralytics
    - args_yaml/overrides: suporta chaves comuns: epochs, batch, imgsz, lr, workers, device, pretrained, num_classes
    """
    cfg = dict(overrides) if overrides is not None else _read_yaml(args_yaml)
    epochs = int(cfg.get("epochs", 10))
    batch = int(cfg.get("batch", 4))
    imgsz = int(cfg.get("imgsz", 640))
    workers = int(cfg.get("workers", 4))
    lr = float(cfg.get("lr", 5e-3))
    pretrained = bool(cfg.get("pretrained", True))

    # Resolver num_classes (+1 background)
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

    device = _device_from_overrides(cfg)

    paths = _resolve_data_yaml_paths(data_yaml)
    train_images, train_labels = paths["train"]
    val_images, val_labels = paths["val"]

    # Option A: não converter para tensor no Albumentations; Dataset fará float32 [0,1]
    train_tf = get_train_transforms(img_size=imgsz, normalize=False, to_tensor=False)
    val_tf = get_val_transforms(img_size=imgsz, normalize=False, to_tensor=False)

    train_ds = AlbumentationsFRCNNDataset(train_images, labels_root=train_labels, transforms=train_tf)
    val_ds = AlbumentationsFRCNNDataset(val_images, labels_root=val_labels, transforms=val_tf) if val_images else None

    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=workers, collate_fn=_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=_collate_fn) if val_ds else None

    model = _build_frcnn(num_classes=num_classes, pretrained=pretrained).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Diretório de saída/checkpoints
    project = cfg.get("project", "runs/frcnn")
    name = cfg.get("name", "exp")
    save_period = int(cfg.get("save_period", -1))
    save_flag = bool(cfg.get("save", True))
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, targets in train_loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        lr_scheduler.step()

        msg = f"[FRCNN] Epoch {epoch+1}/{epochs} - loss: {running_loss/len(train_loader):.4f}"
        if val_loader is not None:
            model.eval()
            with torch.no_grad():
                val_batches = 0
                total_preds = 0
                for images, _ in val_loader:
                    images = [img.to(device) for img in images]
                    outs = model(images)
                    # contagem simples de detecções como sanidade
                    for o in outs:
                        total_preds += int(o.get("boxes", torch.empty(0)).shape[0])
                    val_batches += 1
                msg += f" | val_avg_preds/img: {total_preds/max(1, val_batches*batch):.2f}"
        print(msg)

        # checkpoints por período
        if save_period > 0 and ((epoch + 1) % save_period == 0):
            torch.save(model.state_dict(), str(save_dir / f"weights_epoch{epoch+1}.pth"))

    # checkpoint final
    if save_flag:
        torch.save(model.state_dict(), str(save_dir / "weights_last.pth"))

    return model


def validate_frcnn(
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Validação simples de Faster R-CNN: roda inferência no split=val e retorna as predições brutas.
    """
    cfg = dict(overrides) if overrides is not None else _read_yaml(args_yaml)
    batch = int(cfg.get("batch", 4))
    imgsz = int(cfg.get("imgsz", 640))
    workers = int(cfg.get("workers", 4))
    device = _device_from_overrides(cfg)

    paths = _resolve_data_yaml_paths(data_yaml)
    val_images, val_labels = paths["val"]
    if not val_images:
        raise ValueError("data_yaml não contém 'val'.")

    val_tf = get_val_transforms(img_size=imgsz, normalize=False, to_tensor=False)
    val_ds = AlbumentationsFRCNNDataset(val_images, labels_root=val_labels, transforms=val_tf)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=_collate_fn)

    # reconstruir modelo a partir de weights ou default 2 classes + bg
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


def train_model(
    arch: str,
    data_yaml: str | Path,
    args_yaml: Optional[str | Path] = None,
    weights: Optional[str | Path] = None,
    overrides: Optional[Dict[str, Any]] = None,
):
    """
    Dispatcher: 'v8'/'v11' (YOLO) ou 'frcnn' (Faster R-CNN).
    """
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
