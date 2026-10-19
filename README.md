# 👁️ Sentinela Vision — Monitoramento de Segurança com IA

Sistema autônomo de monitoramento de câmeras de segurança RTSP com detecção e rastreamento de pessoas em tempo real por inteligência artificial (YOLOv8 + ByteTrack), otimizado para aceleração via GPU dedicada NVIDIA.

---

## 🏛️ Arquitetura do Sistema

```text
Cerberus/
├── app/
│   ├── api/
│   │   ├── routes.py          # Rotas REST (/api/cameras, /api/events, /api/stats)
│   │   └── websocket.py       # Endpoints WebSocket (/ws/stream/{cam_id}, /ws/events)
│   ├── core/
│   │   ├── config.py          # Configurações Pydantic e resolução de hardware
│   │   └── logging.py         # Configuração de logs estruturados
│   ├── detector/
│   │   ├── engine.py          # Inferência YOLO com precisão mista FP16 e overlay
│   │   └── tracker.py         # Rastreamento ByteTrack e captura de evidências
│   ├── stream/
│   │   ├── capture.py         # Leitor RTSP desacoplado (zero latência e auto-reconexão)
│   │   └── manager.py         # Orquestrador central de captura e inferência
│   ├── storage/
│   │   └── database.py        # Banco SQLite de histórico com índices otimizados
│   ├── static/
│   │   ├── css/style.css      # Interface Dark Glassmorphism moderna
│   │   ├── js/app.js          # Lógica frontend, WebSockets e renderização Canvas
│   │   └── index.html         # Página principal da aplicação
│   └── main.py                # Entrada FastAPI com lifespan assíncrono
├── config/
│   └── cameras.json           # Cadastro das câmeras RTSP e limiares de confiança
├── data/
│   └── events/                # Recortes e snapshots das pessoas detectadas (.jpg)
├── docker/
│   ├── Dockerfile             # Container CUDA 12.1 runtime
│   └── docker-compose.yml     # Orquestração com GPU pass-through e SELinux (:z)
├── Docs/
│   └── VISION_SPEC.md         # Especificação detalhada de engenharia
├── .env.example
├── requirements.txt
└── README.md
```

---

## ⚙️ Configurações e Variáveis de Ambiente

As configurações podem ser definidas via `.env` ou variáveis de ambiente:

| Variável | Padrão | Descrição |
| :--- | :--- | :--- |
| `PORT` | `8095` | Porta HTTP do servidor web |
| `HOST` | `0.0.0.0` | Host de escuta |
| `DEVICE` | `auto` | Dispositivo de inferência (`cuda:0`, `mps`, `cpu` ou `auto`) |
| `CUDA_VISIBLE_DEVICES` | `0` | Índice da GPU NVIDIA no servidor com Dual-GPU |
| `LOG_LEVEL` | `INFO` | Nível de logs do sistema |
| `MODEL_PATH` | `models/yolov8n.pt` | Caminho do modelo YOLO pré-treinado |
| `CONFIG_PATH` | `config/cameras.json` | Arquivo JSON com lista de câmeras |
| `DATA_DIR` | `data` | Diretório de armazenamento de dados e eventos |
| `CONFIDENCE_THRESHOLD` | `0.65` | Limiar mínimo de confiança para detecção |
| `IOU_THRESHOLD` | `0.45` | Limiar de Non-Maximum Suppression (NMS) |

---

## 💻 Hardware Alvo e Compatibilidade

- **Servidor de Produção (Nobara Linux 44 / Fedora base)**:
  - **GPU Dedicada**: NVIDIA GeForce GTX 1660 (6 GB VRAM) isolada exclusivamente para inferência com suporte a precisão mista FP16 (`half=True`).
  - **SELinux**: Mapeamento de volumes com a flag `:z` configurada no Docker Compose.
  - **Dual-GPU**: AMD RX 5600 XT para o display do sistema e NVIDIA GTX 1660 via `CUDA_VISIBLE_DEVICES=0`.
- **Ambiente de Desenvolvimento (MacBook Air M4 / macOS)**:
  - O código realiza fallback transparente para Apple Silicon Metal (`mps`) ou `cpu`, permitindo testes e desenvolvimento local contínuo sem quebras.

---

## 📡 Endpoints Disponíveis

### REST
- `GET /`: Interface web de monitoramento.
- `GET /api/cameras`: Status de conexão, resolução e FPS das câmeras.
- `GET /api/events`: Histórico paginado de detecções.
- `GET /api/events/{id}/crop`: Imagem recortada da pessoa detectada (com margem de 10%).
- `GET /api/events/{id}/full`: Frame completo com anotações e bounding boxes.
- `GET /api/stats`: Telemetria do sistema e uso de VRAM/hardware.

### WebSockets
- `WS /ws/stream/{camera_id}`: Stream contínuo de frames JPEG binários.
- `WS /ws/events`: Canal de notificações instantâneas de novas pessoas detectadas.
