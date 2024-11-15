import cv2
import numpy as np
from ultralytics import YOLO
import time
import torch
import threading

# Путь к модели светофоров, линии и машин (сегментационная модель для машин)
traffic_light_model_path = 'C:\\Python\\pythonProject\\trafic_lights_detection-3\\runs\\detect\\TLD\\weights\\best.pt'
line_model_path = 'C:\\Python\\pythonProject\\DriveLensAI1-7\\runs\\segment\\FastSAM-x.pt\\weights\\best.pt'
vehicle_model_path = "yolov8n.pt"  # Предобученная YOLOv8 модель для распознавания машин

# Загрузка моделей
traffic_light_model = YOLO(traffic_light_model_path)
line_model = YOLO(line_model_path)
vehicle_model = YOLO(vehicle_model_path)

# Перенос моделей на GPU (если доступен)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
traffic_light_model.to(device)
line_model.to(device)
vehicle_model.to(device)
print(f"Using device: {device}")

# Создаем постоянное черно-белое изображение для накопления линий
persistent_bw_frame = None

# Флаг для состояния светофора
is_red_light = False

# Словарь для отслеживания следов машин
vehicle_trails = {}

# Функция для обработки кадра с тремя моделями
def process_frame(frame):
    global persistent_bw_frame, is_red_light, vehicle_trails

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
                # Отрисовка машин как белых боксов в черно-белом окне
                cv2.rectangle(bw_frame, (x1, y1), (x2, y2), (255, 255, 255), 2)  # Белый контур

                # Вычисляем центр бокса
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Рисуем белую точку в центре бокса
                cv2.circle(bw_frame, (center_x, center_y), 5, (255, 255, 255), -1)  # Белая точка

                # Отслеживаем траекторию машины
                vehicle_id = f"{x1}_{y1}_{x2}_{y2}"  # Простой способ создания уникального ID для машины
                if vehicle_id not in vehicle_trails:
                    vehicle_trails[vehicle_id] = []  # Сохраняем пустой список для новой машины
                vehicle_trails[vehicle_id].append((center_x, center_y))  # Добавляем текущую позицию (центр)

                # Рисуем следы для машины (белая линия между позициями)
                if len(vehicle_trails[vehicle_id]) > 1:
                    for i in range(1, len(vehicle_trails[vehicle_id])):
                        pt1 = vehicle_trails[vehicle_id][i - 1]
                        pt2 = vehicle_trails[vehicle_id][i]
                        cv2.line(bw_frame, pt1, pt2, (255, 255, 255), 2)  # Белая линия

                # Разделяем рамку на три части для проверки пересечения с линией
                rect_height = y2 - y1
                top_part_y2 = y1 + rect_height // 3  # Верхняя граница первой трети
                middle_part_y1 = top_part_y2  # Верхняя граница второй трети
                middle_part_y2 = y1 + 2 * (rect_height // 3)  # Нижняя граница второй трети
                bottom_part_y1 = middle_part_y2  # Верхняя граница третьей трети


                if class_name == "truck":
                    middle_mask = np.zeros_like(persistent_bw_frame, dtype=np.uint8)
                    cv2.rectangle(middle_mask, (x1, middle_part_y1), (x2, middle_part_y2), 255, -1)

                    intersection = cv2.bitwise_and(persistent_bw_frame, middle_mask)
                    intersection_area = np.sum(intersection == 255)
                    middle_area = (x2 - x1) * (middle_part_y2 - middle_part_y1)
                    if middle_area > 0:
                        overlap_percentage = (intersection_area / middle_area) * 100
                        if overlap_percentage > 15 and is_red_light:
                            cv2.putText(bw_frame, "Violation!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)
                            print("Salom")
                            cv2.putText(bw_frame, "Salom", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255), 2)
                elif class_name == "car":
                    bottom_mask = np.zeros_like(persistent_bw_frame, dtype=np.uint8)
                    cv2.rectangle(bottom_mask, (x1, bottom_part_y1), (x2, y2), 255, -1)

                    intersection = cv2.bitwise_and(persistent_bw_frame, bottom_mask)
                    intersection_area = np.sum(intersection == 255)
                    bottom_area = (x2 - x1) * (y2 - bottom_part_y1)
                    if bottom_area > 0:
                        overlap_percentage = (intersection_area / bottom_area) * 100
                        if overlap_percentage > 2 and is_red_light:
                            cv2.putText(bw_frame, "Violation!", (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 2)
                            print("Salom")
                            cv2.putText(bw_frame, "Salom", (x1, y1 - 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (255, 255, 255), 2)
    # Комбинируем постоянное изображение линий с текущим черно-белым кадром
    combined_bw_frame = cv2.bitwise_or(persistent_bw_frame, bw_frame)

    return frame, combined_bw_frame

# Захват видео
video_path = "C:\\Python\\pythonProject\\Traffic-Light-1\\report_1313352.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
start_time = time.time()

# Оптимизация FPS: обрабатывать один кадр из 2 (или больше)
frame_skip = 4

# Функция для обработки видео в отдельном потоке
def process_video():
    global frame_count
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Пропускаем кадры
        if frame_count % frame_skip != 0:
            continue

        # Обработка текущего кадра
        frame, combined_bw_frame = process_frame(frame)

        # Вывод FPS на экран
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Показываем результат
        cv2.imshow("Frame", frame)
        cv2.imshow("Black and White View", combined_bw_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Запуск обработки видео в отдельном потоке
video_thread = threading.Thread(target=process_video)
video_thread.start()

# Ожидание завершения потока
video_thread.join()

cap.release()
cv2.destroyAllWindows()
