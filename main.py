import cv2
from src.setup_gui import SetupGUI
from src.process_video import process_video

CONFIG = {
    'video_path': './data/videos/video7.mp4',
    'model_path': './data/models/yolov8n.pt',
    'classes_path': './config/classes.txt',
    'target_classes': ['car', 'truck', 'bus', 'motorcycle'],
    'confidence_threshold': 0.3,
    'output_resolution': (848, 480), # Alterar a resolução conforme necessário (baseado no vídeo de exemplo)
    'tracker': {
        'max_age': 60, # Aumentado para lidar melhor com oclusões
        'min_hits': 3,
        'iou_threshold': 0.3
    }
}

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