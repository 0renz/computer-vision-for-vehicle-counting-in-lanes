import cv2
import cvzone
import numpy as np
from ultralytics import YOLO
from sort import Sort

# =============================================================================
# 1. CONFIGURAÇÕES GERAIS
# =============================================================================
CONFIG = {
    'video_path': './videos/video7.mp4',
    'model_path': './models/yolov8n.pt',
    'classes_path': './classes.txt',
    'target_classes': ['car', 'truck', 'bus', 'motorcycle'],
    'confidence_threshold': 0.3,
    'output_resolution': (848, 480),
    'tracker': {
        'max_age': 60, # Aumentado para lidar melhor com oclusões
        'min_hits': 3,
        'iou_threshold': 0.3
    }
}

# =============================================================================
# 2. CLASSE PARA GERENCIAR A LÓGICA DO PERCURSO (CHECKPOINTS)
# =============================================================================
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

# =============================================================================
# 3. CLASSE PARA DETECÇÃO E RASTREAMENTO DE VEÍCULOS
# =============================================================================
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

# =============================================================================
# 4. CLASSE PARA A INTERFACE GRÁFICA DE CONFIGURAÇÃO
# =============================================================================
class SetupGUI:
    """Gerencia a janela e a interatividade para desenhar os percursos."""
    def __init__(self, frame):
        self.base_frame = frame.copy()
        self.window_name = 'Config. Percursos - (N)ovo Percurso | (S)tart | (R)eset | (Q)uit'
        self.collected_paths = []
        self.temp_points = []
        
    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.temp_points.append((x, y))
            print(f"Ponto adicionado: ({x}, {y})")

    def run(self):
        """Inicia o loop da GUI para o usuário desenhar os percursos."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n--- Iniciando Configuração de Percursos ---")
        print(" - Clique múltiplos pontos para desenhar os 'checkpoints' de um percurso.")
        print(" - Pressione 'N' para finalizar o percurso atual e começar um novo.")
        print(" - Pressione 'R' para apagar os pontos do percurso que está sendo desenhado.")
        print(" - Pressione 'S' para iniciar a detecção com os percursos definidos.")
        print(" - Pressione 'Q' para sair.")
        
        while True:
            display_frame = self.base_frame.copy()
            if len(self.temp_points) > 1:
                cv2.polylines(display_frame, [np.array(self.temp_points, np.int32)], isClosed=False, color=(255, 0, 0), thickness=1)
            for pt in self.temp_points:
                cv2.circle(display_frame, pt, 5, (0, 0, 255), -1)

            cv2.imshow(self.window_name, display_frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): self.collected_paths = []; print("Configuração cancelada."); break
            elif key == ord('r'): print("Percurso atual resetado."); self.temp_points = []
            elif key == ord('n'):
                if len(self.temp_points) >= 2:
                    self.collected_paths.append(list(self.temp_points))
                    print(f"Percurso {len(self.collected_paths)} finalizado. Pronto para o próximo.")
                    cv2.polylines(self.base_frame, [np.array(self.temp_points, np.int32)], isClosed=False, color=(0, 255, 0), thickness=1)
                    self.temp_points = []
                else: print("Um percurso precisa de pelo menos 2 pontos.")
            elif key == ord('s'):
                if len(self.temp_points) >= 2:
                    self.collected_paths.append(list(self.temp_points))
                if not self.collected_paths: print("Nenhum percurso definido.")
                else: print("\n--- Configuração Concluída ---"); break
        
        cv2.destroyAllWindows()
        return self.collected_paths

# =============================================================================
# 5. FUNÇÃO PRINCIPAL DE PROCESSAMENTO DE VÍDEO
# =============================================================================
def process_video(config, defined_paths_coords, target_w, target_h):
    """Loop principal que processa o vídeo frame a frame."""
    cap = cv2.VideoCapture(config['video_path'])
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {config['video_path']}")
        return

    # Inicializa o rastreador
    tracker = VehicleTracker(config['model_path'], config['classes_path'], config['target_classes'], config['confidence_threshold'], config['tracker'])
    
    # Inicializa os percursos
    paths = {}
    zone_colors = [((0,255,0),(0,255,255)), ((255,0,0),(255,255,0)), ((255,0,255),(0,0,255)), ((255,165,0),(255,215,0))]
    for i, path_points in enumerate(defined_paths_coords):
        path_name = chr(ord('A') + i)
        normal_color, counted_color = zone_colors[i % len(zone_colors)]
        paths[path_name] = PathZone(path_name, path_points, normal_color, counted_color)

    vehicle_path_progress = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Fim do vídeo ou erro de leitura.")
            break
        
        frame_resized = cv2.resize(frame, (target_w, target_h))
        
        tracked_results = tracker.track_vehicles(frame_resized)
        
        current_tracked_ids = set()
        for result in tracked_results:
            x1, y1, x2, y2, obj_id = map(int, result)
            current_tracked_ids.add(obj_id)
            for path in paths.values():
                if path.process_vehicle(obj_id, (x1, y1, x2, y2), vehicle_path_progress):
                    print(f"Veículo ID {obj_id} completou o percurso '{path.name}'!")

        # Limpeza do progresso de veículos que sumiram
        ids_to_remove = set(vehicle_path_progress.keys()) - current_tracked_ids
        for old_id in ids_to_remove:
            del vehicle_path_progress[old_id]

        # Desenha as informações no frame
        y_offset = 40
        for path in paths.values():
            path.draw(frame_resized, vehicle_path_progress)
            cvzone.putTextRect(frame_resized, f'Percurso {path.name}: {path.counter}', (30, y_offset), 2, 1, border=1, colorR=path.normal_color)
            y_offset += 40

        cv2.imshow('Contagem por Cobertura de Percurso', frame_resized)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("\n--- Contagem Final ---")
    for path in paths.values(): print(f"Total no Percurso {path.name}: {path.counter}")

# =============================================================================
# 6. PONTO DE ENTRADA PRINCIPAL
# =============================================================================
def main():
    """Orquestra as fases de setup e processamento da aplicação."""
    # Fase 1: Obter o primeiro frame para a GUI de configuração
    cap_setup = cv2.VideoCapture(CONFIG['video_path'])
    if not cap_setup.isOpened():
        print(f"Erro fatal ao abrir vídeo: {CONFIG['video_path']}")
        return
    ret, first_frame = cap_setup.read()
    if not ret:
        print("Erro fatal ao ler o primeiro frame.")
        cap_setup.release()
        return
    cap_setup.release()

    # Calcula as dimensões de trabalho mantendo a proporção
    max_w, max_h = CONFIG['output_resolution']
    h_orig, w_orig = first_frame.shape[:2]
    ratio = min(max_w / w_orig, max_h / h_orig)
    final_w, final_h = int(w_orig * ratio), int(h_orig * ratio)
    setup_frame = cv2.resize(first_frame, (final_w, final_h))

    # Fase 2: Executar a GUI de configuração para o usuário desenhar os percursos
    gui = SetupGUI(setup_frame)
    defined_paths = gui.run()

    # Fase 3: Se o usuário definiu percursos, iniciar o processamento do vídeo
    if defined_paths:
        process_video(CONFIG, defined_paths, final_w, final_h)
    else:
        print("Nenhum percurso foi definido. Encerrando.")

if __name__ == "__main__":
    main()