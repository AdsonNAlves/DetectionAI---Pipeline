from fastapi import FastAPI
from ultralytics import YOLO
from PIL import Image
import requests
import io
import os
import torch
import torchvision
import cv2
import numpy as np
import base64
import time
import json
from fastapi.responses import StreamingResponse

app = FastAPI()

MODEL_PATHS = {
    "yolo8": os.getenv("YOLO8_PATH", "/app/models/best_yolo8m.pt"),
    "yolo11": os.getenv("YOLO11_PATH", "/app/models/best_yolo11m.pt"),
    "frcnn": os.getenv("FRCNN_PATH", "/app/models/frcnn.pth"),
}
# meta_path = "app/models/metadata.json"

meta_path = os.path.join(os.path.dirname(__file__), "models/metadata.json")

device = "cuda" if torch.cuda.is_available() else "cpu"
loaded_models = {}

def get_model(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modelo '{model_name}' não está disponível. Opções: {list(MODEL_PATHS.keys())}")
    if model_name not in loaded_models:
        if model_name.startswith("yolo"):
            loaded_models[model_name] = YOLO(MODEL_PATHS[model_name])
        elif model_name == "frcnn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
            num_classes = 2
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, num_classes
            )
            state = torch.load(MODEL_PATHS[model_name], map_location=device)
            model.load_state_dict(state)
            model.to(device)
            model.eval()
            loaded_models[model_name] = model
    return loaded_models[model_name]

@app.get("/predict/imgjson")
async def predict_image_json(url: str, model: str = "yolo11", report_id: int = 1, confidence: float = 0.8):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        return {"error": f"Erro ao baixar imagem: {str(e)}"}

    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    mdl = get_model(model)

    start_time = time.time()

    detections = []
    if model.startswith("yolo"):
        results = mdl(image,conf=confidence)
        plotted = results[0].plot()
        for i, box in enumerate(results[0].boxes):
            
            detections.append({
                "id": int(i),
                "class_id": int(box.cls),
                "class_name": "broca" if mdl.names[int(box.cls)]=='item' else mdl.names[int(box.cls)],
                #"class_name": results[0].names[int(box.cls)],
                "confidence": float(box.conf),
                "bbox": {
                    "xmin": float(box.xyxy[0][0]),
                    "ymin": float(box.xyxy[0][1]),
                    "xmax": float(box.xyxy[0][2]),
                    "ymax": float(box.xyxy[0][3]),
                }
            })
    else:
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image)
        outputs = mdl([img_tensor.to(device)])[0]
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for i, (box, score, label) in enumerate(zip(outputs["boxes"], outputs["scores"], outputs["labels"])):
            score_val = float(score.item())
            if score_val >= confidence:
                x1, y1, x2, y2 = box.int().tolist()
                detections.append({
                    "id": int(i),
                    "class_id": int(label.item()),
                    "class_name": "broca" if str(label.item())=='1' else str(label.item()),
                    "confidence": score_val,
                    "bbox": {"xmin": x1, "ymin": y1, "xmax": x2, "ymax": y2}
                })
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        plotted = img_np

    inference_time_ms = int((time.time() - start_time) * 1000)

    # img_pil = Image.fromarray(plotted[..., ::-1]) if model.startswith("yolo") else Image.fromarray(plotted)
    # img_bytes = io.BytesIO()
    # img_pil.save(img_bytes, format="PNG")
    # img_base64 = base64.b64encode(img_bytes.getvalue()).decode("utf-8")


    with open(meta_path, "r") as f:
        metadata_json = json.load(f)
    
    quant_detect = {}
    for det in detections:
        class_name = det["class_name"]
        quant_detect[class_name] = quant_detect.get(class_name, 0) + 1
    
    model_data= metadata_json.get(model)
    
    return {
        "report_id": report_id,
        "image_url": url,
        #"image_det": f"data:image/png;base64,{img_base64}",
        "detections": detections,
        "metadata": {
            "model": model,
            "model_version": model_data["model_version"],
            "classes_model": model_data["classes"],
            "classes_detect": set([d["class_name"] for d in detections]),
            "input_size": [image.size],
            "bbox_format": model_data["bbox_format"],
        },
        "confidence_min": confidence,
        "quant_detect": quant_detect,
        "inference_time_ms": inference_time_ms,
    }


@app.get("/predict/imgpng")
async def annotate_imgpng(url: str, report_id: int = 1, model: str = "yolo11", confidence: float = 0.8):
    response = requests.get(url, timeout=10)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    mdl = get_model(model)
    results = mdl(image, conf=confidence)
    plotted = results[0].plot()  # numpy array BGR

    # Converter para bytes PNG
    img_pil = Image.fromarray(cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB))
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
