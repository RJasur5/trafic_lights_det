from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 model (or a custom model path like "yolo11x-seg.pt")
    model = YOLO('yolov8n.pt')  # Replace with 'yolov8-seg.pt' if starting from scratch

    # Train the model
    model.train(
        data='C:\\Python\\pythonProject\\trafic_lights_detection-3\\data.yaml',  # Path to your dataset configuration file
        epochs=50,                 # Set the number of epochs you want
        imgsz=640,                  # Image size for training
        batch=16,                   # Batch size
        name='TLD'          # Name of the training session
    )
