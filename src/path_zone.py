import numpy as np
import cv2

class PathZone:
    """Gerencia um único percurso, seu estado e a lógica de contagem de veículos."""
    def __init__(self, name, path_points, normal_color, counted_color):
        if len(path_points) < 2:
            raise ValueError("Um percurso precisa de pelo menos 2 pontos.")
        self.name = name
        self.points = np.array(path_points, np.int32)
        self.total_checkpoints = len(self.points)
        self.counter = 0
        self.processed_ids = set()
        self.normal_color = normal_color
        self.counted_color = counted_color
        self.current_color = normal_color

    def process_vehicle(self, object_id, bounding_box, vehicle_progress_data):
        """Verifica se a bounding box de um veículo cobre o próximo checkpoint."""
        if object_id in self.processed_ids:
            return False

        if object_id not in vehicle_progress_data:
            vehicle_progress_data[object_id] = {}
        
        next_checkpoint_index = vehicle_progress_data[object_id].get(self.name, 0)
        
        if next_checkpoint_index >= self.total_checkpoints:
            return False

        target_checkpoint = self.points[next_checkpoint_index]
        x1, y1, x2, y2 = bounding_box
        px, py = target_checkpoint

        if x1 <= px <= x2 and y1 <= py <= y2:
            new_progress = next_checkpoint_index + 1
            vehicle_progress_data[object_id][self.name] = new_progress
            
            if new_progress == self.total_checkpoints:
                self.counter += 1
                self.processed_ids.add(object_id)
                self.current_color = self.counted_color
                if self.name in vehicle_progress_data[object_id]:
                    del vehicle_progress_data[object_id][self.name]
                return True
        return False

    def draw(self, frame, vehicle_progress_data=None):
        """Desenha o percurso e seus checkpoints no frame."""
        cv2.polylines(frame, [self.points], isClosed=False, color=self.normal_color, thickness=1)
        for i, point in enumerate(self.points):
            checkpoint_color = (0, 0, 255) # Vermelho (padrão)
            if vehicle_progress_data:
                for progress_dict in vehicle_progress_data.values():
                    if progress_dict.get(self.name, 0) > i:
                        checkpoint_color = (0, 255, 0) # Verde (concluído)
                        break
            cv2.circle(frame, tuple(point), 7, checkpoint_color, -1)
            cv2.putText(frame, str(i+1), (point[0]+5, point[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

    def reset_color(self):
        self.current_color = self.normal_color