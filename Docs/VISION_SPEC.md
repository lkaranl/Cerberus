# 👁️ Especificação de Engenharia — Sentinela Vision (Sistema de Monitoramento com Detecção de Pessoas via IA)

> **Instruções para a IA executora**: Este documento é a especificação arquitetural e funcional completa para a criação de um **novo projeto independente** de monitoramento de segurança com inteligência artificial, utilizando aceleração por GPU dedicada. Siga rigorosamente a estrutura modular, boas práticas e os parâmetros aqui definidos.

---

## 🎯 1. Objetivo do Projeto

Criar uma aplicação autônoma de **monitoramento de câmeras de segurança RTSP** com **detecção de pessoas em tempo real por IA**, aproveitando a aceleração por hardware de uma GPU **NVIDIA GeForce GTX 1660 (6 GB VRAM)**.

O sistema deve fornecer:
1. **Interface Web Moderna de Monitoramento**:
   - Alternância fluida entre **Modo Câmera Única (Single View)** e **Modo Mosaico (Grid View)**.
   - Visualização das transmissões de vídeo ao vivo com caixas delimitadoras (*Bounding Boxes*) desenhadas sobre as pessoas detectadas, indicando percentual de confiança e identificador persistente (*Tracking ID*).
2. **Pipeline de Inferência em GPU (CUDA)**:
   - Uso de modelos modernos YOLO (YOLOv8 ou YOLOv11 versão Nano/Small) rodando na GTX 1660.
   - Rastreamento contínuo (*Object Tracking*) para evitar alertas duplicados da mesma pessoa.
3. **Histórico de Eventos e Captura de Presença**:
   - Registro de detecções com timestamp, nome da câmera, foto recortada (*crop*) da pessoa e thumbnail da cena.
4. **Resiliência a Quedas de Rede**:
   - Gerenciamento robusto de conexões RTSP com auto-reconexão e tolerância a oscilações Wi-Fi/interferências de RF.

---

## 💻 2. Infraestrutura e Hardware Alvo (Servidor Real)

- **Sistema Operacional**: **Nobara Linux 44** (KDE Plasma Desktop Edition, base Fedora 44, Kernel Linux 7.2.x, Btrfs com SELinux).
- **Processador (CPU)**: Intel(R) Core(TM) i5-7600K (4 núcleos @ 4.40 GHz).
- **Memória RAM**: 32 GB DDR4 (amplo espaço para múltiplos buffers e fila de frames).
- **Configuração de Vídeo (Dual-GPU)**:
  - **GPU 1**: AMD Radeon RX 5600 XT (responsável pelo display/área de trabalho Wayland).
  - **GPU 2**: **NVIDIA GeForce GTX 1660 (6 GB GDDR5, Turing TU116, 1408 CUDA Cores)**:
    - **Driver NVIDIA**: `595.104.02` (suporte a CUDA até `13.2`).
    - **VRAM Total**: 6144 MiB (praticamente **100% livre e desimpedida**, sem carga de interface gráfica).
    - **PCI Bus ID**: `0000:04:00.0` (`Disp.A: Off`).
    - **Papel**: **DEDICADA EXCLUSIVAMENTE PARA A INFERÊNCIA DE IA/CUDA**.
- **Atenção Técnica para a IA (Setup Dual-GPU & SELinux)**:
  - Como o servidor possui **duas placas de vídeo distintas** (AMD + NVIDIA), o runtime NVIDIA no Docker (`nvidia-container-toolkit`) isolará automaticamente a GTX 1660 para o container como dispositivo `cuda:0`.
  - Caso a aplicação seja executada nativamente fora do container, é obrigatório definir `export CUDA_VISIBLE_DEVICES="0"` (ou o respectivo índice da placa NVIDIA) para que o PyTorch não entre em conflito com a placa AMD.
  - Por se tratar de uma distribuição baseada em Fedora/Nobara com SELinux ativo, todos os mapeamentos de volumes no Docker devem utilizar a flag `:z` (ex: `../data:/app/data:z`) para conceder permissão de escrita segura.
- **Stack Tecnológica Recomendada**:
  - **Linguagem**: Python 3.11+.
  - **Framework Web**: **FastAPI** (assíncrono, alta performance, OpenAPI/Swagger embutido).
  - **Servidor ASGI**: Uvicorn com suporte a WebSockets.
  - **Engine de IA**: **PyTorch com CUDA 12.x** e **Ultralytics YOLO** (`yolov8n.pt` ou `yolov11n.pt`).
  - **Processamento de Vídeo**: OpenCV (`opencv-python-headless`) ou PyAV / FFmpeg com leitura assíncrona em thread separada para descarte de frames defasados (evita acúmulo de atraso/buffer).
  - **Frontend**: HTML5, CSS3 moderno (Dark Glassmorphism) e Vanilla JavaScript (sem dependências de frameworks pesados no cliente, suporte a WebSockets nativo).

---

## 📷 3. Câmeras Configuradas e Credenciais

O sistema deve vir pré-configurado com as seguintes câmeras RTSP:

| ID | Nome de Exibição | URL RTSP Completa | Prioridade |
| :--- | :--- | :--- | :--- |
| `cam-0` | **Câmera do Portão** | `rtsp://etbt:dx4ana@192.168.68.53:554/stream1` | Alta (Portão / Entrada) |
| `cam-1` | **Câmera da Garagem** | `rtsp://afey:dm24t4@192.168.68.57:554/stream1` | Normal |
| `cam-2` | **Câmera Área do Fundo**| `rtsp://wnsj:7kr3kw@192.168.68.56:554/stream1` | Normal |

### Configuração em Formato JSON (`config/cameras.json`):
```json
[
  {
    "id": "cam-0",
    "name": "Câmera do Portão",
    "url": "rtsp://etbt:dx4ana@192.168.68.53:554/stream1",
    "active": true,
    "confidence_threshold": 0.65,
    "priority": true
  },
  {
    "id": "cam-1",
    "name": "Câmera da Garagem",
    "url": "rtsp://afey:dm24t4@192.168.68.57:554/stream1",
    "active": true,
    "confidence_threshold": 0.65,
    "priority": false
  },
  {
    "id": "cam-2",
    "name": "Câmera Área do Fundo",
    "url": "rtsp://wnsj:7kr3kw@192.168.68.56:554/stream1",
    "active": true,
    "confidence_threshold": 0.65,
    "priority": false
  }
]
```

---

## 🏛️ 4. Estrutura de Diretórios Recomendada

Adotar arquitetura limpa em camadas:

```text
sentinela-vision/
├── app/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # Rotas REST (/api/cameras, /api/events, /api/settings)
│   │   └── websocket.py       # Endpoints WebSocket (/ws/stream/{cam_id})
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py          # Configurações Pydantic (device CUDA, cam configs, portas)
│   │   └── logging.py         # Configuração de logs estruturados
│   ├── detector/
│   │   ├── __init__.py
│   │   ├── engine.py          # Inicialização e inferência YOLO em GPU (CUDA)
│   │   └── tracker.py         # Rastreamento com ByteTrack e persistência de IDs
│   ├── stream/
│   │   ├── __init__.py
│   │   ├── capture.py         # Leitor RTSP em thread isolada (buffer=1 frame sem delay)
│   │   └── manager.py         # Orquestrador dos streams de todas as câmeras ativas
│   ├── storage/
│   │   ├── __init__.py
│   │   └── database.py        # SQLite / JSON para histórico de eventos de presença
│   ├── static/
│   │   ├── css/
│   │   │   └── style.css      # Interface dark mode premium
│   │   ├── js/
│   │   │   └── app.js         # Lógica da interface, WebSocket, Grid e Single view
│   │   └── index.html         # Página principal da aplicação
│   └── main.py                # Ponto de entrada FastAPI (lifespan, middlewares, router)
├── data/
│   ├── events/                # Recortes e snapshots das pessoas detectadas (.jpg)
│   └── events.db              # Banco SQLite de histórico
├── docker/
│   ├── Dockerfile             # Multi-stage com suporte a NVIDIA CUDA
│   └── docker-compose.yml     # Orquestração com GPU pass-through
├── models/
│   └── yolov8n.pt             # Modelo pré-treinado (~6MB)
├── config/
│   └── cameras.json           # Cadastro das câmeras e credenciais
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚡ 5. Requisitos do Pipeline de Inteligência Artificial

1. **Seleção de Dispositivo**:
   - O código deve inicializar explicitamente verificando a disponibilidade de GPU:
     ```python
     import torch
     device = "cuda:0" if torch.cuda.is_available() else "cpu"
     ```
   - Na GTX 1660, utilizar precisão mista FP16 (`half=True`) para ganho de até 2x na taxa de quadros e menor uso de VRAM (< 1.5 GB).
2. **Filtragem Estrita de Classes**:
   - Apenas a classe `0` (`person` do dataset COCO) deve ser processada e desenhada. Veículos, animais ou objetos devem ser ignorados.
3. **Limiar de Confiança e NMS**:
   - Limiar de confiança configurável (padrão `conf=0.65`).
   - Non-Maximum Suppression (`iou=0.45`).
4. **Rastreamento de Objetos (Object Tracking)**:
   - Utilizar o tracker integrado da Ultralytics (`model.track(frame, persist=True, tracker="bytetrack.yaml")`).
   - Cada pessoa recebe um identificador numérico único (ex: `ID #3`).
   - Um evento de presença só é registrado uma única vez por `ID` enquanto a pessoa estiver na cena.
5. **Captura de Evidência**:
   - Ao confirmar uma nova pessoa na cena por mais de 2 frames consecutivos:
     - Salvar a imagem recortada (*crop*) da pessoa com margem de 10%.
     - Salvar o frame completo com a caixa desenhada.
     - Registrar no banco de dados: `{ id, camera_id, timestamp, tracking_id, confidence, crop_path, full_path }`.

---

## 📺 6. Requisitos da Interface de Monitoramento (UI / UX)

1. **Modos de Exibição**:
   - **Modo Grade / Mosaico (Grid View)**: Exibe as 3 câmeras em layout responsivo (2x2 ou 1x3). Todas as transmissões ativas simultaneamente com suas caixas de detecção.
   - **Modo Tela Única (Single View)**: Permite focar em uma única câmera em tamanho grande, com seletor rápido no topo ou na lateral.
2. **Visualização das Caixas (Overlay)**:
   - Caixas verdes ou ciano com espessura de 2px.
   - Rótulo superior contendo: `Pessoa #ID (89%)`.
3. **Painel Lateral de Alertas em Tempo Real**:
   - Lista cronológica reversa (mais recentes no topo).
   - Card de alerta exibindo:
     - Miniatura (*crop*) do rosto/corpo da pessoa.
     - Nome da câmera e horário exato.
     - Clique no card abre modal com a foto em alta resolução.
4. **Indicadores de Status por Câmera**:
   - Badge `ONLINE` / `OFFLINE` em tempo real.
   - Medidor de FPS de inferência em cada janela de vídeo.

---

## 🔄 7. Resiliência de Conexão RTSP (Anti-Congelamento)

Para evitar os problemas de travamento e latência acumulada típicos de streams RTSP em câmeras Wi-Fi:

1. **Thread de Captura Dedicada (*Zero Latency Worker*)**:
   - O OpenCV padrão acumula frames em buffer interno quando a inferência demora, causando atraso de 5 a 10 segundos no vídeo.
   - **Solução obrigatória**: Uma thread de leitura contínua com `cap.grab()` descartando frames antigos e mantendo apenas o último frame em uma fila de tamanho 1 (`queue.Queue(maxsize=1)`).
2. **Auto-Reconexão**:
   - Se a leitura falhar por mais de 3 segundos (timeout de leitura):
     - Liberar o objeto `cv2.VideoCapture`.
     - Aguardar 2 segundos.
     - Reinstanciar a conexão RTSP via TCP (`os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"`).

---

## 🐳 8. Configuração Docker com Suporte a GPU NVIDIA

### `docker/Dockerfile`:
```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8095

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8095"]
```

### `docker/docker-compose.yml`:
```yaml
version: '3.8'

services:
  sentinela-vision:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    container_name: sentinela-vision-app
    restart: unless-stopped
    ports:
      - "8095:8095"
    volumes:
      # A flag ':z' é mandatória no Nobara/Fedora por conta do SELinux
      - ../data:/app/data:z
      - ../config:/app/config:ro,z
    environment:
      - PORT=8095
      - DEVICE=cuda:0
      - CUDA_VISIBLE_DEVICES=0
      - LOG_LEVEL=INFO
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### `requirements.txt`:
```text
fastapi>=0.110.0
uvicorn[standard]>=0.28.0
pydantic>=2.6.0
pydantic-settings>=2.2.0
ultralytics>=8.1.0
torch>=2.2.0
torchvision>=0.17.0
opencv-python-headless>=4.9.0
numpy>=1.26.0
aiofiles>=23.2.1
python-multipart>=0.0.9
```

---

## 🚀 9. Endpoints Mínimos da API REST & WebSocket

- `GET /`: Interface Web completa (HTML/CSS/JS).
- `GET /api/cameras`: Lista de câmeras, status atual e FPS.
- `GET /api/events`: Histórico das últimas detecções de pessoas com paginação.
- `GET /api/events/{id}/crop`: Download da imagem recortada da pessoa.
- `GET /api/events/{id}/full`: Download do frame completo da detecção.
- `WS /ws/stream/{camera_id}`: Stream contínuo de frames JPEG já com caixas anotadas (ou envio de frames binários + telemetria JSON).

---

## 💡 10. Checklist de Entrega para a IA Desenvolvedora

- [ ] Criação de projeto Python modular seguindo a estrutura proposta.
- [ ] Inicialização da GPU NVIDIA GeForce GTX 1660 com PyTorch e YOLOv8/v11.
- [ ] Leitor RTSP em thread desacoplada com transporte forçado em TCP para as 3 câmeras fornecidas.
- [ ] Tracker ByteTrack configurado apenas para a classe `person`.
- [ ] Interface Web com botões de alternância: **Grid Mosaico** e **Câmera Única**.
- [ ] Painel lateral com cards de alerta exibindo miniaturas das pessoas detectadas.
- [ ] Dockerfile e Docker Compose configurados com reserva da GPU NVIDIA.
