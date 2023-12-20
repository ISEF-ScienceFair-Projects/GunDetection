from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model.train(data='Scifair-1/data.yaml', epochs=100, imgsz=640, device='mps')

model._save_to_state_dict("yolov8x_model_state.pt")