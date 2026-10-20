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
- `GET /`: Interface web de monitoramento ao vivo.
- `GET /dashboard`: Painel analítico e estatístico completo (gráficos por horário, câmeras e pessoas).
- `GET /api/cameras`: Status de conexão, resolução e FPS das câmeras.
- `GET /api/events`: Histórico paginado de detecções.
- `GET /api/events/{id}/crop`: Imagem recortada da pessoa detectada (com margem de 10%).
- `GET /api/events/{id}/full`: Frame completo com anotações e bounding boxes.
- `GET /api/stats`: Telemetria do sistema e uso de VRAM/hardware.
- `GET /api/stats/analytics`: Dados estatísticos agregados para gráficos e relatórios.
- `GET /api/faces`: Lista de pessoas conhecidas carregadas na memória.
- `POST /api/faces/reload`: Recarrega fotos da pasta `data/known_faces/` sem reiniciar o servidor.

### WebSockets
- `WS /ws/stream/{camera_id}`: Stream contínuo de frames JPEG binários.
- `WS /ws/events`: Canal de notificações instantâneas de novas pessoas detectadas.

---

## 🚀 Como Usar Pela Primeira Vez (Execução Nativa no Linux)

### 1. Instalar Pré-requisitos do Sistema
No Nobara / Fedora Linux, garanta a presença do FFmpeg e bibliotecas de desenvolvimento do Python:
```bash
sudo dnf install -y ffmpeg python3-devel
```

### 2. Criar e Ativar o Ambiente Virtual
No diretório raiz do projeto:
```bash
python3 -m venv .venv

# Se você usa Bash ou Zsh:
source .venv/bin/activate

# Se você usa Fish Shell:
source .venv/bin/activate.fish
```

### 3. Instalar as Dependências
Com o ambiente virtual ativado:
```bash
pip install -r requirements.txt
pip install --no-deps facenet-pytorch
```

### 4. Validar o Reconhecimento da GPU NVIDIA
Execute este comando de teste para confirmar que o PyTorch se conecta com a GTX 1660 via CUDA:
```bash
python3 -c "import torch; print('CUDA Disponível:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### 5. Configurar o Arquivo de Ambiente
Gere o arquivo `.env` a partir do modelo padrão:
```bash
cp .env.example .env
```

---

## 🚀 Execução Autônoma 24/7 (Serviço systemd)

Para que o Cerberus continue rodando ininterruptamente em segundo plano mesmo após fechar a sessão SSH ou reiniciar o servidor, utilize o instalador automático:

```bash
./setup_service.sh
```

### O que o script faz automaticamente:
1. Detecta o diretório do projeto, o usuário atual e o Python do ambiente virtual (`.venv`).
2. Gera e instala o arquivo de serviço `/etc/systemd/system/cerberus.service`.
3. Recarrega as configurações do `systemd` e habilita o serviço para iniciar no boot do Linux.
4. Inicia o serviço e verifica a integridade de execução.
5. Identifica o **IP real da sua rede local** e exibe as URLs prontas para acesso.

### Comandos Úteis de Gerenciamento do Serviço:
```bash
# Ver status em tempo real
systemctl status cerberus

# Acompanhar logs ao vivo com rolagem automática
journalctl -u cerberus -f

# Reiniciar o serviço (após alterar configurações ou atualizar código)
sudo systemctl restart cerberus

# Parar o serviço temporariamente
sudo systemctl stop cerberus
```

---

## 🛠️ Execução Manual (Modo Desenvolvimento / Debug)

Caso prefira rodar no terminal com o ambiente virtual ativado:
```bash
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8095
```

---

## 🌐 Acesso ao Painel Web

Abra o navegador no endereço retornado pelo script (ou substitua pelo IP da sua máquina na rede):
- **Painel de Monitoramento ao Vivo**: `http://<ip-do-servidor>:8095`
- **Dashboard Estatístico e Analítico**: `http://<ip-do-servidor>:8095/dashboard`
- **No próprio servidor**: `http://localhost:8095`
