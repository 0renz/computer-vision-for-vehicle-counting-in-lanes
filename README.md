# Computer Vision for Vehicle Counting in Lanes

Este repositório implementa um sistema de visão computacional para detectar, rastrear e contar veículos em vídeos de tráfego, usando detecção por deep learning e rastreamento por múltiplos objetos. O fluxo principal permite ao usuário desenhar caminhos (lanes/zonas de contagem) em um frame inicial e, em seguida, o sistema realiza a contagem automaticamente durante a reprodução do vídeo.

## Visão geral do projeto

O sistema combina três etapas principais:

- Detecção de objetos em cada frame com YOLOv8;
- Rastreamento dos veículos entre frames com SORT;
- Contagem por trajetória, definida manualmente por pontos desenhados pelo usuário.

A ideia central é identificar veículos relevantes (carro, caminhão, ônibus e motocicleta) e contabilizar quantos deles atravessam um percurso previamente definido na cena.

## Objetivo

O projeto foi desenvolvido para automatizar a contagem de veículos em fluxos vidosos de trânsito, com foco em cenários em que é necessário monitorar a passagem por corredores ou faixas específicas. Ele pode ser útil para análise de tráfego, planejamento viário, melhorias operacionais e estudos acadêmicos em visão computacional.

## Arquitetura e fluxo de execução

O ponto de entrada é o arquivo principal:

- `main.py`: configura o vídeo, carrega os parâmetros, abre a interface inicial para definição de percursos e inicia o processamento.

A pipeline do sistema segue esta lógica:

1. O vídeo é carregado.
2. O primeiro frame é exibido para configuração.
3. O usuário desenha um ou mais percursos clicando em vários pontos na imagem.
4. Ao confirmar, o processamento do vídeo começa.
5. Cada frame passa por detecção de veículos com YOLOv8.
6. O rastreador SORT vincula os veículos entre os frames.
7. Cada caminho definido é validado por checkpoints e incrementa a contagem quando um veículo percorre a sequência completa.
8. O vídeo processado é exibido em tempo real com contadores por percurso.

## Estrutura do repositório

```text
.
├── README.md
├── main.py
├── requirements.txt
├── config/
│   └── classes.txt
├── data/
│   ├── models/
│   │   └── yolov8n.pt
│   └── videos/
│       └── (vídeos de entrada)
├── src/
│   ├── get_video_resolution.py
│   ├── path_zone.py
│   ├── process_video.py
│   ├── setup_gui.py
│   ├── sort.py
│   └── vehicle_tracker.py
└── .gitignore
```

## Componentes principais

### `main.py`

Arquivo responsável por:

- definir a configuração do vídeo e do modelo;
- localizar o primeiro frame para preparar a GUI;
- chamar `SetupGUI` para desenho dos percursos;
- iniciar o processamento principal em `process_video`.

Configuração principal do projeto:

```python
CONFIG = {
    'video_path': './data/videos/teste-3.mp4',
    'model_path': './data/models/yolov8n.pt',
    'classes_path': './config/classes.txt',
    'target_classes': ['car', 'truck', 'bus', 'motorcycle'],
    'confidence_threshold': 0.3,
    'output_resolution': NATIVE_RES,
    'tracker': {
        'max_age': 60,
        'min_hits': 3,
        'iou_threshold': 0.3
    }
}
```

Isso significa que o sistema está configurado para detectar automaticamente veículos de interesse com confiança mínima de 30%.

### `src/setup_gui.py`

Implementa a interface interativa de configuração de percursos. Através da janela OpenCV, o usuário pode:

- clicar em vários pontos para construir um caminho;
- pressionar `N` para finalizar um percurso;
- pressionar `R` para reiniciar os pontos do percurso atual;
- pressionar `S` para confirmar e iniciar a análise do vídeo;
- pressionar `Q` para cancelar.

Esse desenho define a trajetória que será usada para contagem. Cada percurso é armazenado como uma sequência de checkpoints.

### `src/path_zone.py`

Representa um percurso individual. Cada `PathZone` contém:

- nome do percurso (`A`, `B`, `C`, ...);
- sequência de pontos do caminho;
- contador total de veículos que completaram o percurso;
- conjunto de IDs já processados;
- lógica para verificar se o veículo cobriu o próximo checkpoint.

A lógica de contagem funciona do seguinte modo:

- para cada veículo rastreado, o sistema verifica seu bounding box;
- se o bounding box cobre o próximo ponto do trajeto, o progresso é atualizado;
- quando o veículo alcança o último checkpoint do percurso, a contagem é incrementada;
- o veículo é marcado como processado para evitar dupla contagem.

### `src/vehicle_tracker.py`

Classe central para detectar e rastrear veículos. Ela faz o seguinte:

- carrega o modelo YOLOv8 em `self.model`;
- lê os nomes das classes em `classes.txt`;
- filtra apenas as classes desejadas (`car`, `truck`, `bus`, `motorcycle`);
- usa a biblioteca `Sort` para associar detectados entre frames e atribuir IDs estáveis.

Trecho principal:

```python
results = self.model(frame, stream=True, verbose=False)
for res in results:
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        confidence, cls_index = float(box.conf[0]), int(box.cls[0])

        if class_name in self.target_classes and confidence > self.conf_threshold:
            detections = np.vstack((detections, [x1, y1, x2, y2, confidence]))
```

### `src/sort.py`

Este arquivo contém a implementação do rastreador SORT (Simple Online and Realtime Tracking), adaptado para o projeto. Ele é responsável por:

- prever a posição atual dos objetos;
- associar deteções novas com rastreios existentes via IoU;
- manter IDs consistentes ao longo do vídeo;
- fornecer a estrutura necessária para contar objetos em múltiplos frames.

### `src/process_video.py`

É o módulo de processamento do vídeo. Ele:

- abre o vídeo de entrada;
- instancia o `VehicleTracker`;
- transforma os percursos desenhados em objetos `PathZone`;
- percorre cada frame;
- aplica detecção + rastreamento;
- atualiza os estados dos percursos;
- desenha contadores e caminhos no frame;
- exibe a janela final em tempo real.

A lógica central é:

```python
tracked_results = tracker.track_vehicles(frame_resized)
for result in tracked_results:
    x1, y1, x2, y2, obj_id = map(int, result)
    for path in paths.values():
        if path.process_vehicle(obj_id, (x1, y1, x2, y2), vehicle_path_progress):
            print(f"Veículo ID {obj_id} completou o percurso '{path.name}'!")
```

### `src/get_video_resolution.py`

Módulo auxiliar para obter a largura e a altura do vídeo, preservando o tamanho da resolução original em configurações do projeto.

## Detecção e classes suportadas

As classes de interesse estão em `config/classes.txt` e incluem objetos padrão do COCO, mas o sistema foi configurado para priorizar:

- `car`
- `truck`
- `bus`
- `motorcycle`

A lista completa do arquivo inclui, por exemplo, pessoas, bicicletas, ônibus, caminhões, bicicletas, animais e outros objetos, mas a filtragem por `target_classes` reduz a análise ao contexto de tráfego.

## Dependências

O projeto usa as seguintes bibliotecas principais:

- `opencv-python`
- `ultralytics`
- `cvzone`
- `numpy`

O arquivo `requirements.txt` contém a configuração básica do ambiente:

```txt
ultralytics
cvzone
opencv-python
numpy
```

## Pré-requisitos

Antes de executar, é necessário:

- Python 3.9+;
- ambiente virtual recomendado;
- cópia do modelo YOLOv8 em `data/models/yolov8n.pt`;
- vídeo de entrada em `data/videos/`.

> O arquivo `yolov8n.pt` já está presente na estrutura do projeto, mas em alguns ambientes pode ser necessário baixar a versão correspondente do Ultralytics antes da execução.

## Como executar

1. Clone o repositório:

```bash
git clone https://github.com/0renz/computer-vision-for-vehicle-counting-in-lanes.git
cd computer-vision-for-vehicle-counting-in-lanes
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Verifique se o vídeo e o modelo existem:

```text
data/videos/teste-3.mp4
data/models/yolov8n.pt
```

5. Execute a aplicação:

```bash
python main.py
```

## Como usar a interface

Ao abrir a aplicação:

- a primeira tela mostra um frame do vídeo para configuração;
- clique nos pontos que representam o início do trajeto;
- continue clicando para montar a linha do percurso;
- pressione `N` para fechar esse percurso e começar outro;
- pressione `S` para iniciar a contagem;
- pressione `Q` para sair.

Durante a execução, o programa exibe a janela com:

- contadores por percurso;
- linha do caminho;
- bounding boxes dos veículos detectados;
- identificação dos checkpoints alcançados;
- total final por rota ao fim do processamento.

## Observações técnicas

- O sistema trabalha com uma abordagem de detecção por frame e rastreamento por associação temporal.
- A contagem não depende apenas da presença do veículo na cena, mas da sequência definida por pontos do percurso.
- O algoritmo foi desenhado para evitar contagem duplicada pela mesma trajetória.
- O projeto usa OpenCV para visualização e manipulação de frames, e YOLOv8 para detecção de objetos.

## Possíveis extensões futuras

- exportar relatórios em CSV ou JSON;
- salvar o vídeo processado em arquivo;
- adicionar suporte a múltiplas câmeras;
- incluir contagem por direção de movimento;
- criar interface mais robusta com seleção de regiões e parâmetros em tempo real;
- permitir diferentes modelos YOLOv8 ou versões treinadas para cenário específico.

## Conclusão

Este projeto reúne visão computacional, rastreamento de múltiplos objetos e lógica de contagem por zona para criar uma solução prática de análise de tráfego em vídeos. A combinação de YOLOv8 + SORT + contagem por percurso torna o sistema útil para estudos, protótipos e aplicações de monitoramento de mobilidade urbana.
