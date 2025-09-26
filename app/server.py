from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import requests
import io
import os
import torch
import torchvision
import cv2
import numpy as np

app = FastAPI()

MODEL_PATHS = {
    "yolo8": os.getenv("YOLO8_PATH", "/app/models/best_yolo8m.pt"),
    "yolo11": os.getenv("YOLO11_PATH", "/app/models/best_yolo11m.pt"),
    "frcnn": os.getenv("FRCNN_PATH", "/app/models/frcnn.pth"),
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# Cache de modelos 
loaded_models = {}

def get_model(model_name: str):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Modelo '{model_name}' não está disponível. Opções: {list(MODEL_PATHS.keys())}")
    if model_name not in loaded_models:
        # YOLO usa ultralytics, FRCNN precisa ser tratado separadamente
        if model_name.startswith("yolo"):
            loaded_models[model_name] = YOLO(MODEL_PATHS[model_name])
        elif model_name == "frcnn":
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
            num_classes = 2  # ajuste pro seu dataset
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

@app.get("/predict")
async def predict(url: str, model: str = "yolo8",request_id: int = 1):
    """
    Faz predição em uma imagem a partir de uma URL.
    Parâmetros:
      - url: URL da imagem
      - model: 'yolo8', 'yolo11' ou 'frcnn'
    """
    # 1. Baixa a imagem com timeout
    try:
        response = requests.get(url, timeout=10)
        if response.status_code != 200:
            return {"error": f"Erro ao baixar imagem da URL: {url}, status_code: {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Erro ao baixar imagem: {str(e)}"}

    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    # 2. Selecionar modelo
    mdl = get_model(model)

    # 3. Fazer predição
    if model.startswith("yolo"):
        results = mdl(image)
        plotted = results[0].plot()  # numpy (BGR)
    else:
        import torchvision.transforms as T
        transform = T.Compose([T.ToTensor()])
        img_tensor = transform(image)
        # outputs = mdl([img_tensor])[0] 
        outputs = mdl([img_tensor.to(device)])[0]

        # desenhar bounding boxes manualmente
        img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        for box, score, label in zip(outputs["boxes"], outputs["scores"], outputs["labels"]):
            if score > 0.5:  # threshold
                x1, y1, x2, y2 = box.int().tolist()
                cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img_np, f"{label.item()}:{score:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        plotted = img_np

    # 4. Converter para PNG
    img_pil = Image.fromarray(plotted[..., ::-1]) if model.startswith("yolo") else Image.fromarray(plotted)
    img_bytes = io.BytesIO()
    img_pil.save(img_bytes, format="PNG")
    img_bytes.seek(0)

    return StreamingResponse(img_bytes, media_type="image/png")
