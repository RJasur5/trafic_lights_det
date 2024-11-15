import cv2
import numpy as np
import os
from ultralytics import YOLO
import time

# Путь к модели светофоров, линии и машин (сегментационная модель для машин)
traffic_light_model_path = 'C:\\Python\\pythonProject\\trafic_lights_detection-3\\runs\\detect\\TLD\\weights\\best.pt'
line_model_path = 'C:\\Python\\pythonProject\\DriveLensAI1-7\\runs\\segment\\FastSAM-x.pt\\weights\\best.pt'
vehicle_model_path = "yolov8n.pt"  # Предобученная YOLOv8 модель для распознавания машин

# Загрузка моделей
traffic_light_model = YOLO(traffic_light_model_path)
line_model = YOLO(line_model_path)
vehicle_model = YOLO(vehicle_model_path)  # Используем предобученную YOLOv8 для распознавания машин

# Создаем постоянное черно-белое изображение для накопления линий
persistent_bw_frame = None

# Флаг для состояния светофора
is_red_light = False

# Параметры буфера для хранения кадров
frame_buffer = []
violation_detected = False
violation_start_frame = 0
frame_count = 0

# Путь для сохранения видео
output_folder = "C:\\Python\\pythonProject\\trafic_lights_detection-3"
# Проверка существования папки, если нет - создаем
if not os.path.exists(output_folder):
    os.makedirs(output_folder)


# Функция для генерации уникального имени для файла
def get_unique_video_path(output_folder, base_filename="violation_video.mp4"):
    # Получаем путь к файлу
    output_video_path = os.path.join(output_folder, base_filename)

    # Проверка существования файла, если существует - добавляем уникальный идентификатор
    if os.path.exists(output_video_path):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        base_filename = f"violation_video_{timestamp}.mp4"
        output_video_path = os.path.join(output_folder, base_filename)

    return output_video_path


# Получаем уникальный путь для видео
output_video_path = get_unique_video_path(output_folder)

# Создание видеопотока для сохранения кадров
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None


# Функция для обработки кадра с тремя моделями
def process_frame(frame, frame_count, cap):
    global persistent_bw_frame, is_red_light, frame_buffer, violation_detected, violation_start_frame, out

    # Изменяем размер кадра на 1080x720
    frame = cv2.resize(frame, (1080, 720))

    # Инициализируем постоянное черно-белое изображение, если оно еще не создано
    if persistent_bw_frame is None:
        persistent_bw_frame = np.zeros_like(frame, dtype=np.uint8)

    # Обнаружение светофоров для основного окна
    traffic_results = traffic_light_model(frame)
    is_red_light = False  # Сброс состояния светофора перед проверкой
    for result in traffic_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = traffic_light_model.names[cls]

            print(f"Detected traffic light class: {class_name}, Confidence: {conf:.2f}")

            if "Red" in class_name:  # Если "Red" присутствует в названии класса
                print("Красный свет горит")
                is_red_light = True
            elif "Green" in class_name:
                print("Зеленый свет горит")

            # Отрисовка рамки для светофора
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Создаем черно-белое изображение для текущего кадра
    bw_frame = np.zeros_like(frame, dtype=np.uint8)

    # Обнаружение дорожных линий
    line_results = line_model(frame)
    for result in line_results:
        masks = result.masks
        if masks is not None:
            for mask, cls in zip(masks, result.boxes.cls):
                if line_model.names[int(cls)] == 'random-line':  # Отображаем только 'random-line'
                    mask_img = mask.data.cpu().numpy()[0]
                    mask_img = cv2.resize(mask_img, (frame.shape[1], frame.shape[0]))
                    persistent_bw_frame[mask_img > 0.5] = (255, 255, 255)  # Добавляем линии на постоянное изображение

    # Обнаружение машин с использованием предобученной YOLOv8
    vehicle_results = vehicle_model(frame)
    for result in vehicle_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = vehicle_model.names[cls]

            # Фильтруем только автомобили и грузовики
            if class_name == "car" or class_name == "truck":
                # Отрисовка рамки для машин на основном окне
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Отрисовка машин как белых боксов в черно-белом окне
                cv2.rectangle(bw_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Белый контур

                # Разделяем рамку на три части
                rect_height = y2 - y1
                top_part_y2 = y1 + rect_height // 3  # Верхняя граница первой трети
                middle_part_y1 = top_part_y2  # Верхняя граница второй трети
                middle_part_y2 = y1 + 2 * (rect_height // 3)  # Нижняя граница второй трети
                bottom_part_y1 = middle_part_y2  # Верхняя граница третьей трети

                if class_name == "truck":
                    # Создаем маску только для средней трети грузовика
                    middle_mask = np.zeros_like(persistent_bw_frame, dtype=np.uint8)
                    cv2.rectangle(middle_mask, (x1, middle_part_y1), (x2, middle_part_y2), 255, -1)

                    # Вычисляем пересечение средней трети грузовика с сохраненной линией
                    intersection = cv2.bitwise_and(persistent_bw_frame, middle_mask)
                    intersection_area = np.sum(intersection == 255)
                    middle_area = (x2 - x1) * (middle_part_y2 - middle_part_y1)
                    if middle_area > 0:
                        overlap_percentage = (intersection_area / middle_area) * 100
                        # Отображаем процент пересечения над рамкой автомобиля в черно-белом окне
                        cv2.putText(bw_frame, f"{overlap_percentage:.2f}%", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Проверка условия нарушения для средней трети грузовика
                        if overlap_percentage > 15 and is_red_light:
                            # Отображаем надпись "Violation!" над грузовиком
                            cv2.putText(frame, "Violation!", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 2)
                            violation_detected = True
                            violation_start_frame = frame_count

                elif class_name == "car":
                    # Создаем маску только для нижней трети автомобиля
                    bottom_mask = np.zeros_like(persistent_bw_frame, dtype=np.uint8)
                    cv2.rectangle(bottom_mask, (x1, bottom_part_y1), (x2, y2), 255, -1)

                    # Вычисляем пересечение нижней трети автомобиля с сохраненной линией
                    intersection = cv2.bitwise_and(persistent_bw_frame, bottom_mask)
                    intersection_area = np.sum(intersection == 255)
                    bottom_area = (x2 - x1) * (y2 - bottom_part_y1)
                    if bottom_area > 0:
                        overlap_percentage = (intersection_area / bottom_area) * 100
                        # Отображаем процент пересечения над рамкой машины в черно-белом окне
                        cv2.putText(bw_frame, f"{overlap_percentage:.2f}%", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                        # Проверка условия нарушения для нижней трети автомобиля
                        if overlap_percentage > 15 and is_red_light:
                            # Отображаем надпись "Violation!" над машиной
                            cv2.putText(frame, "Violation!", (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 255), 2)
                            violation_detected = True
                            violation_start_frame = frame_count

    # Буфер кадров для сохранения видео с нарушением (с 2 секундами до и после)
    if violation_detected:
        frame_buffer.append((frame, bw_frame))
        if len(frame_buffer) > 4:  # 4 секунды = 4 * 30 кадров (при 30 fps)
            violation_detected = False
            violation_start_frame = -0.5
            # Сохранение или обработка обрезанных кадров
            if out is None:
                out = cv2.VideoWriter(output_video_path, fourcc, 30, (frame.shape[1], frame.shape[0]))
            for frame, bw in frame_buffer:
                out.write(frame)  # Записываем кадры с нарушения

    # Комбинируем постоянное изображение линий с текущим черно-белым кадром
    combined_bw_frame = cv2.bitwise_or(persistent_bw_frame, bw_frame)

    return frame, combined_bw_frame


# Захват видео
video_path = "C:\\Python\\pythonProject\\Traffic-Light-1\\report_1313352.mp4"
cap = cv2.VideoCapture(video_path)

# Счетчик кадров
frame_count = 0

# Обработка видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Проверка, является ли текущий кадр чётным (каждый второй кадр)
    if frame_count % 4 == 0:
        # Обработка каждого второго кадра
        frame, combined_bw_frame = process_frame(frame, frame_count, cap)

        # Отображение исходного кадра и черно-белого окна
        cv2.imshow("Traffic Light and Vehicle Detection", frame)
        cv2.imshow("Black and White View: Persistent Lines and Vehicles", combined_bw_frame)

    frame_count += 1  # Увеличиваем счетчик кадров

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Закрытие видеопотока
cv2.destroyAllWindows()
