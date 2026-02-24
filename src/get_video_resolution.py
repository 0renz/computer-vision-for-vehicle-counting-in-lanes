import os
import cv2

def get_video_resolution(video_path):
    if not os.path.exists(video_path):
        print(f"Erro: O arquivo {video_path} não foi encontrado.")
        return (640, 480) # Fallback padrão

    cap = cv2.VideoCapture(video_path)
    
    # Captura a largura e altura como floats e converte para int
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    # Se por algum motivo retornar 0, envia um padrão seguro
    if width == 0 or height == 0:
        return (640, 480)
        
    return (width, height)