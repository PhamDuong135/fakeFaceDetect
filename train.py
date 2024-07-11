from ultralytics import YOLO

model = YOLO("../models/yolov8n.pt")

model.train(data='Dataset/SplitData/dataOffline.yaml',epochs=5)