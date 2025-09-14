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

# Загрузка первой модели
def load_model1():
    # Исправляем deprecated параметр
    model = models.efficientnet_b0(weights=None)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, len(class_names1))
    # Добавляем weights_only=True для безопасности
    model.load_state_dict(torch.load(model1_path, map_location="cpu", weights_only=True))
    model.eval()
    return model

# Загрузка второй модели (исправленная версия)
def load_model2():
    try:
        # Сначала пробуем загрузить state_dict
        state_dict = torch.load(model2_path, map_location="cpu", weights_only=True)
        
        # Проверяем разные форматы сохранения
        if isinstance(state_dict, dict):
            # Если это словарь с ключом model_state_dict
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            # Если есть другие ключи, возможно это полная модель
            elif "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
        
        # Чистим ключи от префиксов
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # Убираем возможные префиксы
            name = k.replace("efficientnet.", "").replace("module.", "").replace("model.", "")
            new_state_dict[name] = v
        
        # Создаём модель заново
        model = EfficientNet.from_name(efficientnet_version)
        
        # Заменяем последний слой
        num_features = model._fc.in_features
        model._fc = torch.nn.Linear(num_features, len(class_names2))
        
        # Загружаем веса
        model.load_state_dict(new_state_dict, strict=False)
        model.eval()
        
        print("Модель 2 загружена из state_dict")
        return model
        
    except Exception as e:
        print(f"Ошибка загрузки модели 2: {e}")
        # Создаём пустую модель в случае ошибки
        model = EfficientNet.from_name(efficientnet_version)
        model._fc = torch.nn.Linear(model._fc.in_features, len(class_names2))
        model.eval()
        return model

# Загрузка четвертой модели (Rust Detection) - ONNX формат
def load_model4():
    try:
        # Инициализируем ONNX Runtime сессию
        providers = ['CPUExecutionProvider']  # Используем CPU
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        session = ort.InferenceSession(model4_path, providers=providers)
        print(f"Модель 4 загружена (ONNX) с провайдерами: {providers}")
        return session
    except Exception as e:
        print(f"Ошибка загрузки ONNX модели: {e}")
        return None

# Инициализация третьей модели (Roboflow)
def init_model3():
    return InferenceHTTPClient(
        api_url=ROBOFLOW_API_URL,
        api_key=ROBOFLOW_API_KEY
    )

# Загружаем все модели
print("Загрузка моделей...")
model1 = load_model1()
print("Модель 1 загружена")
model2 = load_model2()
print("Модель 2 загружена")
model4_session = load_model4()
print("Модель 4 загружена")
client = init_model3()
print("Модель 3 инициализирована")
print("Все модели готовы к работе!")

# ==================== ТРАНСФОРМАЦИИ ====================

transform1 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform2 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform4 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== ФУНКЦИИ ПРЕДСКАЗАНИЯ ====================

def predict_model1(image_path):
    """Предсказание для первой модели (Car/NotCar)"""
    try:
        img = Image.open(image_path).convert("RGB")
        img_t = transform1(img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model1(img_t)
            probs = torch.softmax(outputs, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()
        
        return class_names1[pred_class], confidence
    except Exception as e:
        print(f"Ошибка в модели 1: {e}")
        return "error", 0.0

def predict_model2(image_path):
    """Предсказание для второй модели (Damage Detection)"""
    try:
        image = Image.open(image_path).convert("RGB")
        img_t = transform2(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model2(img_t)
            # Проверяем форму вывода
            if outputs.dim() == 2 and outputs.size(1) == len(class_names2):
                probs = torch.softmax(outputs, dim=1)[0]
                pred_idx = torch.argmax(probs).item()
                confidence = probs[pred_idx].item()
                return class_names2[pred_idx], confidence
            else:
                print("Неожиданная форма вывода модели 2")
                return "unknown", 0.0
                
    except Exception as e:
        print(f"Ошибка в модели 2: {e}")
        return "error", 0.0

def predict_model3(image_path):
    """Предсказание для третьей модели (Dirt Detection)"""
    try:
        result = client.infer(image_path, model_id=MODEL_ID)
        
        if result and "predictions" in result and result["predictions"]:
            confs = [p["confidence"] for p in result["predictions"]]
            avg_conf = np.mean(confs)
            return avg_conf
        else:
            return 0.0
    except Exception as e:
        print(f"Ошибка в модели 3: {e}")
        return 0.0

def predict_model4(image_path):
    """Предсказание для четвертой модели (Rust Detection) - ONNX"""
    try:
        if model4_session is None:
            return "model_error", 0.0
            
        img = Image.open(image_path).convert("RGB")
        img_t = transform4(img)
        
        # Преобразуем в numpy array и меняем порядок осей для ONNX
        img_np = img_t.numpy().astype(np.float32)
        
        # Получаем имя входного тензора
        input_name = model4_session.get_inputs()[0].name
        
        # Выполняем предсказание
        outputs = model4_session.run(None, {input_name: img_np[np.newaxis, ...]})
        
        # Обрабатываем выходы
        if len(outputs) > 0:
            output = outputs[0]
            probs = torch.softmax(torch.tensor(output), dim=1)
            conf, pred_idx = torch.max(probs, dim=1)
            pred_class = class_names4[pred_idx.item()]
            confidence = conf.item()
            return pred_class, confidence
        else:
            return "no_output", 0.0
            
    except Exception as e:
        print(f"Ошибка в модели 4: {e}")
        return "error", 0.0

def dirt_label_and_color(conf):
    """Функция для определения уровня загрязнения"""
    if conf < 0.1:
        return "very clean", (0, 255, 255)   # cyan
    elif conf < 0.3:
        return "clean", (0, 255, 0)          # green
    elif conf < 0.5:
        return "slightly dirty", (255, 255, 0)  # yellow
    elif conf < 0.7:
        return "dirty", (255, 165, 0)        # orange
    elif conf < 0.9:
        return "very dirty", (255, 0, 0)     # red
    else:
        return "extremely dirty", (128, 0, 128)  # purple

# ==================== ОСНОВНАЯ ФУНКЦИЯ ====================

def analyze_image(image_path):
    """Анализ изображения всеми четырьмя моделями"""
    # Проверяем существование файла
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Файл не найден: {image_path}")
    
    # Загружаем изображение для отображения
    image = cv2.imread(image_path)
    if image is None:
        # Пробуем загрузить через PIL если OpenCV не смог
        pil_image = Image.open(image_path).convert("RGB")
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    display_image = image.copy()
    
    # Получаем предсказания от всех моделей
    print("Анализ изображения...")
    car_pred, car_conf = predict_model1(image_path)
    damage_pred, damage_conf = predict_model2(image_path)
    dirt_conf = predict_model3(image_path)
    rust_pred, rust_conf = predict_model4(image_path)
    dirt_label, dirt_color = dirt_label_and_color(dirt_conf)
    
    # Выводим результаты в консоль
    print("=" * 60)
    print(f"📊 РЕЗУЛЬТАТЫ АНАЛИЗА ИЗОБРАЖЕНИЯ:")
    print(f"🚗 Машина/Не машина: {car_pred} (уверенность: {car_conf:.2f})")
    print(f"🔧 Повреждения: {damage_pred} (уверенность: {damage_conf:.2f})")
    print(f"🧹 Загрязнение: {dirt_label} (уверенность: {dirt_conf:.2f})")
    print(f"🛠️  Ржавчина: {rust_pred} (уверенность: {rust_conf:.2f})")
    print("=" * 60)
    
    # Добавляем текст на изображение
    y_position = 40
    line_height = 35
    
    texts = [
        f"Car: {car_pred} ({car_conf:.2f})",
        f"Damage: {damage_pred} ({damage_conf:.2f})",
        f"Dirt: {dirt_label} ({dirt_conf:.2f})",
        f"Rust: {rust_pred} ({rust_conf:.2f})"
    ]
    
    colors = [(0, 255, 0), (255, 0, 0), dirt_color, (255, 255, 0)]
    
    for i, text in enumerate(texts):
        cv2.putText(display_image, text, (20, y_position + i * line_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
    
    # Отображаем результат
    plt.figure(figsize=(14, 10))
    plt.imshow(display_image)
    plt.axis('off')
    plt.title('Результаты анализа изображения', fontsize=16)
    plt.tight_layout()
    plt.show()
    
    return {
        'car_prediction': car_pred,
        'car_confidence': car_conf,
        'damage_prediction': damage_pred,
        'damage_confidence': damage_conf,
        'dirt_level': dirt_label,
        'dirt_confidence': dirt_conf,
        'rust_prediction': rust_pred,
        'rust_confidence': rust_conf
    }

# ==================== ДИАГНОСТИКА МОДЕЛЕЙ ====================

def diagnose_models():
    """Диагностика всех моделей"""
    print("\n🔍 Диагностика моделей:")
    
    # Модель 2
    print("\nМодель 2:")
    print(f"Тип: {type(model2)}")
    if hasattr(model2, '_fc'):
        print(f"Последний слой: {model2._fc}")
    
    # Модель 4
    print("\nМодель 4:")
    if model4_session:
        print(f"Тип: ONNX InferenceSession")
        print(f"Входы: {[input.name for input in model4_session.get_inputs()]}")
        print(f"Выходы: {[output.name for output in model4_session.get_outputs()]}")
    else:
        print("Модель 4 не загружена")

# ==================== ИСПОЛЬЗОВАНИЕ ====================

if __name__ == "__main__":
    # Подавляем предупреждения
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    
    # Запускаем диагностику моделей
    diagnose_models()
    
    # Укажите путь к вашему изображению
    image_path = input("Введите путь к изображению: ").strip().strip('"')
    
    # Убираем кавычки если они есть
    image_path = image_path.replace('"', '')
    
    try:
        results = analyze_image(image_path)
        print("\n✅ Анализ завершен успешно!")
        
        # Сводка результатов
        print("\n📋 СВОДКА РЕЗУЛЬТАТОВ:")
        if results['car_prediction'] == 'car':
            print("✅ Объект определен как автомобиль")
            print(f"   Состояние: {results['damage_prediction']}")
            print(f"   Чистота: {results['dirt_level']}")
            print(f"   Коррозия: {results['rust_prediction']}")
        else:
            print("❌ Объект не является автомобилем")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        print("Проверьте:")
        print("1. Существует ли файл по указанному пути")
        print("2. Корректность пути к файлу")
        print("3. Доступность файла")
