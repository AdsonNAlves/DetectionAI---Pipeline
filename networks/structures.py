from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from ultralytics import YOLO


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


def _default_weights(arch: str) -> str:
	"""Retorna pesos padrão por arquitetura."""
	arch = arch.lower().strip()
	if arch in {"v8", "yolov8", "yolo8"}:
		return "yolov8m.pt"
	if arch in {"v11", "yolov11", "yolo11"}:
		return "yolo11m.pt"
	raise ValueError("Arquitetura inválida. Use 'v8' ou 'v11'.")


def _load_model(arch: str, weights: Optional[str | Path] = None) -> YOLO:
	"""Carrega um modelo YOLO conforme a arquitetura informada.

	- arch: 'v8' ou 'v11'
	- weights: caminho .pt ou nome de checkpoint (opcional). Se None, usa pesos medium.
	"""
	ckpt = str(weights) if weights else _default_weights(arch)
	return YOLO(ckpt)


def train_yolo(
	arch: str,
	data_yaml: str | Path,
	args_yaml: Optional[str | Path] = None,
	weights: Optional[str | Path] = None,
	overrides: Optional[Dict[str, Any]] = None,
) -> YOLO:
	"""Treina um modelo YOLO.

	Parâmetros:
	- arch: 'v8' ou 'v11'
	- data_yaml: caminho para data.yaml (formato Ultralytics)
	- args_yaml: caminho opcional para args.yaml com overrides, ex.: {epochs, imgsz, batch, device, name, ...}
	- weights: checkpoint inicial (ex.: 'yolov8n.pt', 'yolo11n.pt' ou caminho local)

	Retorna o objeto YOLO carregado (último estado após treino).
	"""
	data_yaml = str(data_yaml)
	if overrides is not None:
		train_kwargs = dict(overrides)
	else:
		train_kwargs = _read_yaml(args_yaml)
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
	"""Valida um modelo YOLO (split=val por padrão).

	- Pode usar args.yaml para overrides (ex.: imgsz, batch, device, conf, iou, etc.).
	- Se 'split' não for definido em args, usa 'val'.
	Retorna métricas/saída do Ultralytics.
	"""
	data_yaml = str(data_yaml)
	overrides = _read_yaml(args_yaml)
	model = _load_model(arch, weights)

	val_kwargs = dict(overrides)
	val_kwargs.setdefault("data", data_yaml)
	val_kwargs.setdefault("split", "val")  # validação

	return model.val(**val_kwargs)


def test_yolo(
	arch: str,
	data_yaml: str | Path,
	args_yaml: Optional[str | Path] = None,
	weights: Optional[str | Path] = None,
) -> Any:
	"""Testa um modelo YOLO (split=test por padrão).

	- Usa o mesmo formato de args do validate/train.
	- Se 'split' não for definido em args, usa 'test'.
	Retorna métricas/saída do Ultralytics.
	"""
	data_yaml = str(data_yaml)
	overrides = _read_yaml(args_yaml)
	model = _load_model(arch, weights)

	test_kwargs = dict(overrides)
	test_kwargs.setdefault("data", data_yaml)
	test_kwargs.setdefault("split", "test")  # teste

	return model.val(**test_kwargs)


__all__ = [
	"train_yolo",
	"validate_yolo",
	"test_yolo",
]

