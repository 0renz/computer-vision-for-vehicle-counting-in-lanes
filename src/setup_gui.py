import cv2
import numpy as np

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