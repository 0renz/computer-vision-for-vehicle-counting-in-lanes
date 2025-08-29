import cv2
import cvzone
from .vehicle_tracker import VehicleTracker
from .path_zone import PathZone

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