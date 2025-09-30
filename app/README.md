# Detection API 

Esta API foi desenvolvida em **FastAPI** para realizar predições de objetos em imagens utilizando modelos de visão computacional:  
- **YOLOv8**  
- **YOLOv11**  
- **Faster R-CNN**  

Os endpoints permitem obter tanto as **detecções em JSON** quanto a **imagem anotada (PNG)**.

---

## 🚀 Endpoints

### 🔹 1. `/predict/imgjson` – Predict Image Json
Retorna as detecções no formato **JSON**.  

**Exemplo de request:**
```bash
curl -X GET "http://localhost:8080/predict/imgjson?url=https://example.com/imagem.jpg&model=yolo11&confidence=0.8"
```
🔹 Request 

```bash
{
    "url": "https://example.com/imagem.jpg",
    "model": yolo11,
    "report_id": 1,
    "confidence": 0.8, 
}
```
🔹 Response

```bash
{
  "report_id": 1,
  "image_url": "https://example.com/imagem.jpg",
  "detections": [
    {
      "id": 0,
      "class_id": 0,
      "class_name": "broca",
      "confidence": 0.92,
      "bbox": {
        "xmin": 34.2,
        "ymin": 56.1,
        "xmax": 128.7,
        "ymax": 200.3
      }
    }
  ],
  "metadata": {
    "model": "yolo11",
    "model_version": "2025_v1",
    "classes_model": ["broca"],
    "classes_detect": ["broca"],
    "input_size": [[640, 640]],
    "bbox_format": "xyxy"
  },
  "confidence_min": 0.8,
  "quant_detect": {
    "broca": 1
  },
  "inference_time_ms": 45
}

```

### 🔹 2. /predict/imgpng – Annotate Image PNG

Retorna a imagem anotada com bounding boxes em formato PNG.  

**Exemplo de request**:  

```bash
curl -X GET "http://localhost:8080/predict/imgpng?url=https://example.com/imagem.jpg&model=yolo11&confidence=0.8" --output resultado.png
```  
Isso salvará o arquivo resultado.png com as detecções desenhadas.  

## ⚙️ Parâmetros dos Endpoints

- `url` → URL da imagem (**obrigatório**)  
- `model` → Modelo a ser usado (`yolo8`, `yolo11`, `frcnn`)  
- `report_id` → ID opcional para rastreabilidade  
- `confidence` → Limiar mínimo de confiança (default: 0.8)  
