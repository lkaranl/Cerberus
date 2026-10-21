# Plano de Implementação: WebRTC com go2rtc e Canvas Overlay

- **Objetivo**: Prover streaming de vídeo com aceleração de hardware a 25–30 FPS e latência sub-80ms via WebRTC (H.264 direto da câmera IP através do go2rtc), sobrepondo caixas delimitadoras e nomes da IA em tempo real com canvas transparente.
- **Arquitetura**: Video Tag (`<video>`) para H.264 acelerado por GPU + Canvas (`<canvas>`) para bounding boxes e crachás biométricos em tempo real via WebSocket de metadados.
- **Referência de Design**: `Docs/superpowers/specs/2026-10-21-webrtc-go2rtc-overlay-design.md`

---

## Estrutura de Arquivos

| Arquivo | Ação | Responsabilidade |
|---|---|---|
| `go2rtc.yaml` | Criar | Configuração das 3 câmeras RTSP para o gateway WebRTC go2rtc |
| `scripts/setup_go2rtc.sh` | Criar | Script de download do binário oficial `go2rtc_linux_amd64` e serviço systemd |
| `app/api/websocket.py` | Modificar | Adicionar rota `/ws/stream-meta/{camera_id}` para transmissão ultraleve de metadados |
| `app/stream/manager.py` | Modificar | Alimentar buffer de metadados mais recente para cada câmera |
| `app/static/index.html` | Modificar | Estruturar cada quadrante com `<video>` e `<canvas>` sobrepostos |
| `app/static/css/style.css` | Modificar | Estilos de posicionamento absoluto do canvas sobre o vídeo |
| `app/static/js/app.js` | Modificar | Cliente WebRTC WHEP com fallback gracioso para WebSocket e desenho de overlays |

---

### Tarefa 1: Configuração do go2rtc e Script de Instalação para Nobara Linux
- **Arquivos**: `go2rtc.yaml`, `scripts/setup_go2rtc.sh`
- **Passos**:
  1. Criar `go2rtc.yaml` mapeando `cam-0`, `cam-1`, `cam-2` com suas respectivas URLs RTSP.
  2. Adicionar configurações de API WHEP na porta `1984` do go2rtc.
  3. Criar `scripts/setup_go2rtc.sh` que baixa a release v1.9.8 do `go2rtc_linux_amd64`, dá permissão de execução e cria o arquivo de serviço `go2rtc.service` no systemd do Linux.

### Tarefa 2: Endpoint WebSocket de Metadados de IA
- **Arquivos**: `app/stream/manager.py`, `app/api/websocket.py`
- **Passos**:
  1. Em `StreamManager`, manter fila/buffer de eventos de metadados por câmera (`self.latest_metadata_queues` ou broadcast).
  2. Em `app/api/websocket.py`, adicionar a rota `@ws_router.websocket("/stream-meta/{camera_id}")`.
  3. Transmitir apenas as detecções atuais enriquecidas com nomes, IDs, caixas em resolução normalizada e rótulos.

### Tarefa 3: Estrutura HTML e Estilos CSS de Dupla Camada (Video + Canvas)
- **Arquivos**: `app/static/index.html`, `app/static/css/style.css`
- **Passos**:
  1. Alterar cada quadrante de câmera para conter `<div class="camera-stream-wrapper">` com `<video id="video-{cam.id}">` e `<canvas id="canvas-{cam.id}">`.
  2. Ajustar CSS para garantir que o canvas fique perfeitamente alinhado por cima do vídeo (`pointer-events: none`).

### Tarefa 4: Cliente WebRTC WHEP e Renderizador de Overlays em JavaScript
- **Arquivos**: `app/static/js/app.js`
- **Passos**:
  1. Implementar função `connectWebRTC(camId, videoEl, onFail)` usando `RTCPeerConnection` e `POST /api/whep?src={camId}`.
  2. Implementar `connectMetadataStream(camId, canvasEl)` para escutar o WebSocket `/ws/stream-meta/{camId}` e pintar as caixas com anti-aliasing e cores neon correspondentes.
  3. Implementar fallback automático: se o WebRTC falhar ou der timeout de 3.5s, voltar para o pipeline WebSocket JPEG padrão de forma transparente.

### Tarefa 5: Commit, Push e Instruções de Execução no Nobara
- **Passos**:
  1. Commit de todas as mudanças em português.
  2. Push para a branch `main`.
  3. Instruções passo a passo para o usuário executar no terminal Nobara via SSH.
