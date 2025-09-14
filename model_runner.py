# model_runner.py
# Adapted from user's script: provides analyze_image(image_path) function used by Flask backend.
import torch
from torchvision import models, transforms
from efficientnet_pytorch import EfficientNet
from inference_sdk import InferenceHTTPClient
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from collections import OrderedDict
import warnings
import os
import torch.nn as nn
import onnxruntime as ort

# ==================== КОНФИГУРАЦИЯ ====================

# Параметры для первой модели (EfficientNet Car/NotCar)
model1_path = r"C:\Users\Arkats\Downloads\efficientnet_car_notcar.pth"
class_names1 = ["car", "not_car"]

# Параметры для второй модели (Damage Detection)
model2_path = r"C:\Users\Arkats\Downloads\best_efficientnet_model.pth"
class_names2 = ["С повреждениями", "Без повреждениями"]
efficientnet_version = "efficientnet-b0"

# Параметры для третьей модели (Roboflow Dirt Detection)
ROBOFLOW_API_URL = "https://detect.roboflow.com"
ROBOFLOW_API_KEY = "xVizONFEb3EZlu1LR8K0"
MODEL_ID = "dirt-detection-meter-b3phd/3"

# Параметры для четвертой модели (Rust Detection) - ONNX формат
model4_path = r"C:\Users\Arkats\Downloads\efficientnet_rust.onnx"
class_names4 = ["clean_car", "rust_car"]

# ==================== ЗАГРУЗКА МОДЕЛЕЙ ====================

def safe_load_torch(path):
    try:
        return torch.load(path, map_location='cpu')
    except Exception as e:
        print('Не удалось загрузить torch файл:', e)
        return None

# Загрузка первой модели
def load_model1():
    try:
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names1))
        state = safe_load_torch(model1_path)
        if state is not None:
            try:
                model.load_state_dict(state)
            except Exception:
                # try if file contains state_dict
                if isinstance(state, dict) and 'state_dict' in state:
                    model.load_state_dict(state['state_dict'])
        model.eval()
        return model
    except Exception as e:
        print('Ошибка при загрузке model1:', e)
        return None

# Загрузка второй модели (исправленная версия)
def load_model2():
    try:
        state_dict = safe_load_torch(model2_path)
        if state_dict is None:
            # Create empty model architecture
            model = EfficientNet.from_name(efficientnet_version)
            model._fc = torch.nn.Linear(model._fc.in_features, len(class_names2))
            model.eval()
            return model
        if isinstance(state_dict, dict):
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace("efficientnet.", "").replace("module.", "").replace("model.", "")
            new_state_dict[name] = v
        model = EfficientNet.from_name(efficientnet_version)
        num_features = model._fc.in_features
        model._fc = torch.nn.Linear(num_features, len(class_names2))
        try:
            model.load_state_dict(new_state_dict, strict=False)
        except Exception as e:
            print('Warning loading model2 state_dict:', e)
        model.eval()
        return model
    except Exception as e:
        print(f"Ошибка загрузки модели 2: {e}")
        model = EfficientNet.from_name(efficientnet_version)
        model._fc = torch.nn.Linear(model._fc.in_features, len(class_names2))
        model.eval()
        return model

# Загрузка четвертой модели (Rust Detection) - ONNX формат
def load_model4():
    try:
        providers = ['CPUExecutionProvider']
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider','CPUExecutionProvider']
        session = ort.InferenceSession(model4_path, providers=providers)
        print(f"ONNX модель загружена с провайдерами: {providers}")
        return session
    except Exception as e:
        print(f"Ошибка загрузки ONNX модели: {e}")
        return None

# Инициализация третьей модели (Roboflow)
def init_model3():
    try:
        return InferenceHTTPClient(api_url=ROBOFLOW_API_URL, api_key=ROBOFLOW_API_KEY)
    except Exception as e:
        print('Ошибка инициализации Roboflow client:', e)
        return None

print('Инициализация моделей (может занять время)...')
model1 = load_model1()
model2 = load_model2()
model4_session = load_model4()
client = init_model3()
print('Инициализация завершена (если модели не найдены, функции вернут заглушки).')

# ==================== ТРАНСФОРМАЦИИ ====================
from torchvision import transforms as T

transform1 = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
transform2 = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
transform4 = T.Compose([T.Resize((224,224)), T.ToTensor(), T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ==================== ФУНКЦИИ ПРЕДСКАЗАНИЯ ====================

def predict_model1(image_path):
    try:
        if model1 is None:
            return 'model_missing', 0.0
        img = Image.open(image_path).convert('RGB')
        img_t = transform1(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model1(img_t)
            probs = torch.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        return class_names1[pred_class], float(confidence)
    except Exception as e:
        print('Ошибка в модели 1:', e)
        return 'error', 0.0

def predict_model2(image_path):
    try:
        if model2 is None:
            return 'model_missing', 0.0
        image = Image.open(image_path).convert('RGB')
        img_t = transform2(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model2(img_t)
            if outputs.dim() == 2 and outputs.size(1) == len(class_names2):
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                return class_names2[pred_idx], float(confidence)
            else:
                return 'unknown', 0.0
    except Exception as e:
        print('Ошибка в модели 2:', e)
        return 'error', 0.0

def predict_model3(image_path):
    try:
        if client is None:
            return 0.0
        result = client.infer(image_path, model_id=MODEL_ID)
        if result and 'predictions' in result and result['predictions']:
            confs = [p.get('confidence',0.0) for p in result['predictions']]
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return avg_conf
        return 0.0
    except Exception as e:
        print('Ошибка в модели 3:', e)
        return 0.0

def predict_model4(image_path):
    try:
        if model4_session is None:
            return 'model_missing', 0.0
        img = Image.open(image_path).convert('RGB')
        img_t = transform4(img)
        img_np = img_t.numpy().astype(np.float32)
        input_name = model4_session.get_inputs()[0].name
        outputs = model4_session.run(None, {input_name: img_np[np.newaxis,...]})
        if len(outputs) > 0:
            output = outputs[0]
            probs = torch.softmax(torch.tensor(output), dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_class = class_names4[pred_idx.item()]
            return pred_class, float(conf.item())
        return 'no_output', 0.0
    except Exception as e:
        print('Ошибка в модели 4:', e)
        return 'error', 0.0

def dirt_label_and_color(conf):
    if conf < 0.1:
        return 'very clean', (0,255,255)
    elif conf < 0.3:
        return 'clean', (0,255,0)
    elif conf < 0.5:
        return 'slightly dirty', (255,255,0)
    elif conf < 0.7:
        return 'dirty', (255,165,0)
    elif conf < 0.9:
        return 'very dirty', (255,0,0)
    else:
        return 'extremely dirty', (128,0,128)

def analyze_image(image_path):
    # Verify file exists
    if not os.path.exists(image_path):
        raise FileNotFoundError(f\"Файл не найден: {image_path}\")
    # Load  image via OpenCV/PIL
    image = cv2.imread(image_path)
    if image is None:
        pil_image = Image.open(image_path).convert('RGB')
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Predictions
    car_pred, car_conf = predict_model1(image_path)
    damage_pred, damage_conf = predict_model2(image_path)
    dirt_conf = predict_model3(image_path)
    rust_pred, rust_conf = predict_model4(image_path)
    dirt_label, dirt_color = dirt_label_and_color(dirt_conf)
    # Build result dict (JSON serializable)
    return {
        'car_prediction': str(car_pred),
        'car_confidence': float(car_conf or 0.0),
        'damage_prediction': str(damage_pred),
        'damage_confidence': float(damage_conf or 0.0),
        'dirt_level': str(dirt_label),
        'dirt_confidence': float(dirt_conf or 0.0),
        'rust_prediction': str(rust_pred),
        'rust_confidence': float(rust_conf or 0.0)
    }
