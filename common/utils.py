from typing import List, Tuple, Dict, Any, Optional
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import torch
import os 

# Transforms padrão com flags para controle de Normalize/ToTensor
def get_train_transforms(img_size: int = 640, *, normalize: bool = True, to_tensor: bool = True) -> A.Compose:
    tfs = [
        # Preserve aspecto e preencha para quadrado
        A.LongestMaxSize(max_size=img_size),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, border_mode=cv2.BORDER_CONSTANT, value=(114,114,114)),
        A.HorizontalFlip(p=0.5),
        # Affine moderado (substitui ShiftScaleRotate + Affine duplicadas)
        A.Affine(scale=(0.95, 1.05), translate_percent=(0.0, 0.05), rotate=(-5, 5), shear=(-3, 3), p=0.5),
        # Cores moderadas
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.4),
        # ---------------
        # A.RandomResizedCrop(img_size, img_size, scale=(0.8, 1.0), p=0.5),
        # A.HorizontalFlip(p=0.7),  # fliplr
        #A.VerticalFlip(p=0.1),    # flipud
        #A.ShiftScaleRotate(
        #    shift_limit=0.2,  # translate
        #    scale_limit=0.5,  # scale
        #    rotate_limit=10,  # degrees
        #    border_mode=0,
        #    p=0.7,
        #),
        #A.Affine(shear=10, p=0.5),  # shear
        #A.ColorJitter(
        #    hue=0.015,         # hsv_h aprox.
        #    saturation=0.7,    # hsv_s
        #    brightness=0.4,    # hsv_v
        #    p=0.7,
        #),
        #A.CoarseDropout(
        #    max_holes=1,       # erasing
        #    max_height=0.4,
        #    max_width=0.4,
        #    min_height=0.1,
        #    min_width=0.1,
        #    p=0.4,
        #),
        #A.Resize(img_size, img_size),
    ]
    # CoarseDropout compatível com diferentes versões do Albumentations
    try:
        tfs.append(A.CoarseDropout(max_holes=1, max_height=int(0.1*img_size), max_width=int(0.1*img_size),
                                   min_holes=1, min_height=1, min_width=1, fill_value=(114,114,114), p=0.15))
    except TypeError:
        tfs.append(A.CoarseDropout(holes=1,
                                   hole_height=(int(0.02*img_size), int(0.1*img_size)),
                                   hole_width=(int(0.02*img_size), int(0.1*img_size)),
                                   fill_value=(114,114,114), p=0.15))
    if normalize:
        tfs.append(A.Normalize())
    if to_tensor:
        tfs.append(ToTensorV2())
    return A.Compose(
        tfs,
        bbox_params=A.BboxParams(
            format="pascal_voc",
            label_fields=["class_labels"],  # mantenha consistente com apply_transforms
            min_visibility=0.05, #0.2,
        ),
    )


def build_transforms_from_args(args: Dict[str, Any], img_size: int = 640) -> A.Compose:
    """Cria um pipeline Albumentations a partir de um dicionário de args.

    Suporta as flags mais comuns que você usava no YOLO args.yaml.
    """
    augs: List[Any] = []

    # Augmentations condicionais (treino)
    augment = bool(args.get("augment", True))
    if augment:
        # crop_fraction -> RandomResizedCrop scale
        crop_fraction = float(args.get("crop_fraction", 0.8))
        augs.append(A.RandomResizedCrop(img_size, img_size, scale=(crop_fraction, 1.0), p=0.6))

        # flips
        fliplr = float(args.get("fliplr", 0.5))
        flipud = float(args.get("flipud", 0.0))
        if fliplr > 0:
            augs.append(A.HorizontalFlip(p=fliplr))
        if flipud > 0:
            augs.append(A.VerticalFlip(p=flipud))

        # geometric
        translate = float(args.get("translate", 0.0))
        degrees = float(args.get("degrees", 0.0))
        shear = float(args.get("shear", 0.0))
        if degrees > 0 or translate > 0 or shear > 0:
            augs.append(
                A.Affine(translate_percent=translate, rotate=degrees, shear=shear, p=0.5)
            )

        # color
        augs.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5))

        # erasing
        erasing = float(args.get("erasing", 0.0))
        if erasing > 0:
            augs.append(A.CoarseDropout(max_holes=1, max_height=int(img_size*erasing), max_width=int(img_size*erasing), p=erasing))

    # always resize + normalize at the end
    augs.append(A.Resize(img_size, img_size))
    augs.append(A.Normalize())

    return A.Compose(augs, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]))

def get_val_transforms(img_size: int = 640, *, normalize: bool = True, to_tensor: bool = True) -> A.Compose:
    tfs = [A.Resize(img_size, img_size)]
    if normalize:
        tfs.append(A.Normalize())
    if to_tensor:
        tfs.append(ToTensorV2())
    return A.Compose(
        tfs,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"]),
    )

# Aplica transform em imagem + bboxes (bboxes: list de [xmin,ymin,xmax,ymax])
def apply_transforms(img: np.ndarray, bboxes: List[List[float]], labels: List[int], transforms: A.Compose):
    if transforms is None:
        return img, bboxes, labels
    data = {"image": img, "bboxes": bboxes, "class_labels": labels}
    out = transforms(**data)
    return out["image"], out["bboxes"], out["class_labels"]

# Converte imagem (H,W,3) numpy para tensor CHW float (0..1)
def img_to_tensor(img: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
    img = np.ascontiguousarray(img)
    img = img.transpose(2, 0, 1).astype(np.float32)  # CHW
    img = torch.from_numpy(img) / 255.0
    if device:
        img = img.to(device)
    return img

# Pós-processa previsões (ex.: boxes em relação ao tamanho original)
def rescale_boxes(boxes: List[List[float]], orig_size: Tuple[int, int], input_size: Tuple[int, int]):
    ow, oh = orig_size
    iw, ih = input_size
    scale_x = ow / iw
    scale_y = oh / ih
    res = []
    for b in boxes:
        xmin, ymin, xmax, ymax = b
        res.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
    return res


# ===== FRCNN utilities: keep structures2.py clean =====
def get_image_hw(img: Any) -> Tuple[int, int]:
    """Return (H, W) from numpy or torch image (HWC or CHW)."""
    if isinstance(img, torch.Tensor):
        if img.ndim == 3:
            if img.shape[0] in (1, 3):  # CHW
                return int(img.shape[1]), int(img.shape[2])
            else:  # HWC
                return int(img.shape[0]), int(img.shape[1])
        raise ValueError("Unsupported tensor image shape: {img.shape}")
    elif isinstance(img, np.ndarray):
        return int(img.shape[0]), int(img.shape[1])
    else:
        raise TypeError("Unsupported image type for get_image_hw: %r" % type(img))


def ensure_tensor_chw_float01(img: Any) -> torch.Tensor:
    """Ensure image is torch.float32, CHW, in [0,1]. Accepts numpy HWC/CHW or torch tensor HWC/CHW (uint8/float)."""
    if isinstance(img, torch.Tensor):
        t = img
        if t.ndim == 3 and t.shape[0] not in (1, 3):  # assume HWC -> CHW
            t = t.permute(2, 0, 1).contiguous()
        if t.dtype != torch.float32:
            t = t.float()
        # normalize to [0,1] if it seems like 0..255
        try:
            if torch.isfinite(t).all() and t.max() > 1:
                t = t / 255.0
        except Exception:
            pass
        return t
    elif isinstance(img, np.ndarray):
        arr = img.astype(np.float32, copy=False)
        if arr.max() > 1.0:
            arr = arr / 255.0
        if arr.ndim == 3 and arr.shape[-1] in (1, 3):  # HWC -> CHW
            arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(arr))
    else:
        raise TypeError("Unsupported image type for ensure_tensor_chw_float01: %r" % type(img))


def clip_and_filter_bboxes(bxs: List[List[float]], lbs: List[int], w: int, h: int) -> Tuple[List[List[float]], List[int]]:
    """Clip boxes within (0,w-1,0,h-1) and drop invalid (non-positive area), keeping labels aligned."""
    nb: List[List[float]] = []
    nl: List[int] = []
    for b, l in zip(bxs, lbs):
        x1 = float(max(0.0, min(b[0], w - 1)))
        y1 = float(max(0.0, min(b[1], h - 1)))
        x2 = float(max(0.0, min(b[2], w - 1)))
        y2 = float(max(0.0, min(b[3], h - 1)))
        if x2 > x1 and y2 > y1:
            nb.append([x1, y1, x2, y2])
            nl.append(int(l))
    return nb, nl

def prepare_frcnn_sample(
    img_np: np.ndarray,
    boxes: List[List[float]],
    labels: List[int],
    transforms: Optional[A.Compose] = None,
    retry_on_empty: bool = True,
    max_tries: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply Albumentations (if any), clip/filter boxes, and return (img_tensor, boxes_t, labels_t)."""
    tries = 0
    img_out, boxes_out, labels_out = img_np, boxes, labels

    while True:
        if transforms is not None and (tries == 0 or (retry_on_empty and tries < max_tries)):
            img_tmp, boxes_tmp, labels_tmp = apply_transforms(img_np, boxes, labels, transforms)
        else:
            img_tmp, boxes_tmp, labels_tmp = img_out, boxes_out, labels_out

        h_cur, w_cur = get_image_hw(img_tmp)
        boxes_tmp, labels_tmp = clip_and_filter_bboxes(boxes_tmp, labels_tmp, w_cur, h_cur)

        if boxes_tmp or not retry_on_empty or tries >= max_tries:
            img_out, boxes_out, labels_out = img_tmp, boxes_tmp, labels_tmp
            break
        tries += 1

    if os.getenv("DEBUG_AUG") == "1":
        before = len(boxes)
        after = len(boxes_out)
        print(f"[AUG] boxes before={before} after={after} tries={tries}")

    img_tensor = ensure_tensor_chw_float01(img_out)

    if len(boxes_out) > 0:
        boxes_t = torch.as_tensor(boxes_out, dtype=torch.float32)
        labels_t = torch.as_tensor(labels_out, dtype=torch.int64)
    else:
        boxes_t = torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.zeros((0,), dtype=torch.int64)

    return img_tensor, boxes_t, labels_t

# def prepare_frcnn_sample(
#     img_np: np.ndarray,
#     boxes: List[List[float]],
#     labels: List[int],
#     transforms: Optional[A.Compose] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """Apply Albumentations (if any), clip/filter boxes, and return (img_tensor, boxes_t, labels_t).

#     - img_np: numpy image HWC RGB
#     - boxes: list of [xmin, ymin, xmax, ymax]
#     - labels: list of int (1..N for torchvision)
#     - transforms: Albumentations compose built with bbox_params matching 'class_labels'
#     """
#     if transforms is not None:
#         img_out, boxes_out, labels_out = apply_transforms(img_np, boxes, labels, transforms)
#     else:
#         img_out, boxes_out, labels_out = img_np, boxes, labels

#     h_cur, w_cur = get_image_hw(img_out)
#     boxes_out, labels_out = clip_and_filter_bboxes(boxes_out, labels_out, w_cur, h_cur)

#     img_tensor = ensure_tensor_chw_float01(img_out)

#     if len(boxes_out) > 0:
#         boxes_t = torch.as_tensor(boxes_out, dtype=torch.float32)
#         labels_t = torch.as_tensor(labels_out, dtype=torch.int64)
#     else:
#         boxes_t = torch.zeros((0, 4), dtype=torch.float32)
#         labels_t = torch.zeros((0,), dtype=torch.int64)

#     return img_tensor, boxes_t, labels_t
