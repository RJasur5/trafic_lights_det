import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import os
import re

# Путь к модели светофоров, линии и машин (сегментационная модель для машин)
traffic_light_model_path = 'C:\\Python\\pythonProject\\trafic_lights_detection-3\\runs\\detect\\TLD\\weights\\best.pt'
line_model_path = 'C:\\Python\\pythonProject\\DriveLensAI1-7\\runs\\segment\\FastSAM-x.pt\\weights\\best.pt'
vehicle_model_path = "yolov8n.pt"  # Предобученная YOLOv8 модель для распознавания машин

# Загрузка моделей
traffic_light_model = YOLO(traffic_light_model_path)
line_model = YOLO(line_model_path)
vehicle_model = YOLO(vehicle_model_path)  # Используем предобученную YOLOv8 для распознавания машин

# Параметры видео
fps = 30  # Частота кадров видео
buffer_seconds = 2  # Количество секунд до и после нарушения
buffer_size = buffer_seconds * fps

# Папка для сохранения нарушений
violations_folder = "violations"
os.makedirs(violations_folder, exist_ok=True)

# Получение следующего доступного индекса для записи нарушений
def get_next_violation_index():
    existing_files = os.listdir(violations_folder)
    indices = [
        int(re.search(r'violation_(\d+)\.jpg', filename).group(1))
        for filename in existing_files if re.search(r'violation_(\d+)\.jpg', filename)
    ]
    return max(indices, default=0) + 1

# Устанавливаем начальный индекс для новых нарушений
violation_index = get_next_violation_index()

# Буфер для хранения последних 2 секунд кадров
frame_buffer = deque(maxlen=buffer_size)

# Флаг и переменные для состояния светофора и записи нарушений
is_red_light = False
violation_frame_countdown = 0  # Счетчик кадров для записи после обнаружения нарушения
out = None  # ВидеоWriter для записи нарушений
record_violation = False  # Флаг для записи видео с нарушением

# Создаем постоянное черно-белое изображение для накопления линий
persistent_bw_frame = None

# Функция для обработки кадра с тремя моделями
def process_frame(frame):
    global persistent_bw_frame, is_red_light, violation_index, violation_frame_countdown, out, record_violation

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

            if "Red" in class_name:  # Если "Red" присутствует в названии класса
                is_red_light = True

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
    violation_detected = False
    violating_boxes = []  # Список для хранения координат нарушающих машин
    for result in vehicle_results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls = int(box.cls[0])
            class_name = vehicle_model.names[cls]

            # Фильтруем только автомобили и грузовики
            if class_name == "car" or class_name == "truck":
                # Отрисовка рамки и метки в реальном времени
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Отрисовка машины в черно-белом окне
                cv2.rectangle(bw_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

                # Разделение на части и проверка на пересечение с линией
                rect_height = y2 - y1
                middle_part_y1 = y1 + rect_height // 3
                middle_part_y2 = y1 + 2 * (rect_height // 3)
                bottom_part_y1 = middle_part_y2

                if class_name == "truck":
                    middle_mask = np.zeros_like(persistent_bw_frame, dtype=np.uint8)
                    cv2.rectangle(middle_mask, (x1, middle_part_y1), (x2, middle_part_y2), 255, -1)
                    intersection = cv2.bitwise_and(persistent_bw_frame, middle_mask)
                    intersection_area = np.sum(intersection == 255)
                    middle_area = (x2 - x1) * (middle_part_y2 - middle_part_y1)
                    if middle_area > 0 and (intersection_area / middle_area) * 100 > 13 and is_red_light:
                        violation_detected = True
                        violating_boxes.append((x1, y1, x2, y2))  # Запоминаем координаты нарушителя
                        # Отображаем надпись "Violation!" над машиной
                        cv2.putText(bw_frame, "Violation!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)

                elif class_name == "car":
                    bottom_mask = np.zeros_like(persistent_bw_frame, dtype=np.uint8)
                    cv2.rectangle(bottom_mask, (x1, bottom_part_y1), (x2, y2), 255, -1)
                    intersection = cv2.bitwise_and(persistent_bw_frame, bottom_mask)
                    intersection_area = np.sum(intersection == 255)
                    bottom_area = (x2 - x1) * (y2 - bottom_part_y1)
                    if bottom_area > 0 and (intersection_area / bottom_area) * 100 > 1.8 and is_red_light:
                        violation_detected = True
                        violating_boxes.append((x1, y1, x2, y2))  # Запоминаем координаты нарушителя
                        # Отображаем надпись "Violation!" над машиной
                        cv2.putText(bw_frame, "Violation!", (x1, y1 - 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)

    # Если нарушение обнаружено, сохраняем изображения всех нарушающих машин
    if violation_detected:
        for violating_box in violating_boxes:
            x1, y1, x2, y2 = violating_box
            violating_image = frame[y1:y2, x1:x2]  # Вырезаем изображение нарушающего автомобиля
            violation_image_path = os.path.join(violations_folder, f"violation_{violation_index}.jpg")
            cv2.imwrite(violation_image_path, violating_image)
            print(f"Нарушение сохранено как {violation_image_path}")
            violation_index += 1

        # Запуск записи видео при обнаружении нарушения
        if not record_violation:
            record_violation = True
            violation_frame_countdown = buffer_size  # Устанавливаем счетчик для записи после нарушения

            # Создание уникального видеофайла для текущего нарушения
            video_path = os.path.join(violations_folder, f"violation_{violation_index}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, fps, (frame.shape[1], frame.shape[0]))

            # Запись кадров из буфера (2 секунды до нарушения)
            for buffered_frame in frame_buffer:
                out.write(buffered_frame)

    # Если идет запись нарушения, добавляем текущий кадр
    if record_violation and out is not None:
        out.write(frame)
        violation_frame_countdown -= 1  # Уменьшаем счетчик

        # Останавливаем запись после заданного количества кадров
        if violation_frame_countdown == 0:
            record_violation = False
            out.release()
            out = None
            print(f"Нарушение сохранено в видео: {video_path}")

    # Если нарушение было обнаружено, отображаем "Violation!" в черно-белом окне
    if violation_detected:
        cv2.putText(bw_frame, "Violation!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)

    # Комбинируем постоянное изображение линий с текущим черно-белым кадром
    combined_bw_frame = cv2.bitwise_or(persistent_bw_frame, bw_frame)

    # Добавление текущего кадра в буфер
    frame_buffer.append(frame.copy())

    return frame, combined_bw_frame


# Захват видео
video_path = "C:\\Python\\pythonProject\\Traffic-Light-1\\report_1313352.mp4"
cap = cv2.VideoCapture(video_path)

# Счетчик кадров
frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % 3 == 0:
        frame, combined_bw_frame = process_frame(frame)

        # Отображение
        cv2.imshow("Traffic Light and Vehicle Detection", frame)
        cv2.imshow("Black and White View: Persistent Lines and Vehicles", combined_bw_frame)

    frame_count += 1

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
