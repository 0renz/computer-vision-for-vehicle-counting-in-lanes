import cv2
import numpy as np
from ultralytics import YOLO
from .sort import Sort

class VehicleTracker:
    """Encapsula o modelo YOLO e o rastreador SORT."""
    def __init__(self, model_path, classes_path, target_classes, conf_threshold, tracker_params):
        self.model = YOLO(model_path)
        try:
            with open(classes_path, 'r') as f:
                self.classnames = f.read().splitlines()
        except FileNotFoundError:
            print(f"Erro: Arquivo de classes '{classes_path}' não encontrado.")
            self.classnames = []
        
        self.target_classes = set(target_classes)
        self.conf_threshold = conf_threshold
        self.tracker = Sort(**tracker_params)

    def track_vehicles(self, frame):
        """Detecta e rastreia veículos em um frame."""
        detections = np.empty((0, 5))
        results = self.model(frame, stream=True, verbose=False)

        for res in results:
            for box in res.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence, cls_index = float(box.conf[0]), int(box.cls[0])
                
                if 0 <= cls_index < len(self.classnames):
                    class_name = self.classnames[cls_index]
                    if class_name in self.target_classes and confidence > self.conf_threshold:
                        detections = np.vstack((detections, [x1, y1, x2, y2, confidence]))
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 150, 0), 2)
        
        return self.tracker.update(detections)