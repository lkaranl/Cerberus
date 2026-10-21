# Especificação de Design: WebRTC de Alta Performance com go2rtc e Canvas Overlay

- **Data**: 21/10/2026
- **Autor**: Antigravity & Karan
- **Componentes**: `go2rtc`, `app/stream/`, `app/api/websocket.py`, `app/static/js/app.js`, `app/static/index.html`
- **Status**: Proposta Aprovada (Opção 2 - Máxima Performance)

---

## 1. Diagnóstico e Motivação

### Problema Atual (WebSocket + JPEG):
1. **Sobrecarga de Rede e I/O**: Cada frame JPEG em 1280x720 possui ~120KB. Com 3 câmeras transmitindo a 25 FPS, o tráfego gerado é de cerca de **90 Mbps** (11 MB/s) contínuos na rede local.
2. **Gargalo de CPU**: O servidor consome tempo precioso codificando JPEGs em software (`cv2.imencode`) dezenas de vezes por segundo.
3. **Limite de Renderização no Navegador**: Decodificar 75 blobs JPEG por segundo via JavaScript/Canvas gera gargalos no garbage collector e limita o frame rate a ~5-15 FPS.

### Solução SOTA (WebRTC H.264 Direto + Overlay de IA):
1. **Consumo Quase Nulo de Rede**: O fluxo H.264 comprimido da câmera é transmitido diretamente via RTP/UDP, consumindo apenas **1.5 a 3 Mbps** para todas as câmeras.
2. **Decodificação Nativa por Hardware**: O navegador decodifica o vídeo diretamente na GPU através da tag `<video>`, liberando 100% da CPU do cliente.
3. **Zero Transcodificação no Servidor**: O servidor não gasta CPU recomprimindo vídeo; o stream original da câmera vai direto para o cliente.
4. **Metadados em Tempo Real**: As detecções de IA (caixas, nomes de moradores, crachás e placas) são enviadas via WebSocket em JSON ultraleve (< 1 KB) e desenhadas por cima do vídeo em um `<canvas>` transparente.
5. **Frame Rate**: Cravado em **20 a 30 FPS** com latência inferior a **80ms**.

---

## 2. Arquitetura do Sistema

```mermaid
flowchart TD
    subgraph Câmeras IP (Tapo)
        Cam0[Câmera Portão - RTSP H.264]
        Cam1[Câmera Garagem - RTSP H.264]
        Cam2[Câmera Fundos - RTSP H.264]
    end

    subgraph Backend Cerberus (Nobara Linux)
        go2rtc[go2rtc Gateway WebRTC]
        IA_Engine[Cerberus AI Pipeline: YOLO11 + ArcFace + OSNet]
        WS_Meta[FastAPI WebSocket: /ws/stream-meta/{cam_id}]
    end

    subgraph Navegador Web (Cliente)
        subgraph Quadrante de Monitoramento
            VideoTag[Tag video: WebRTC H.264 nativo 25 FPS]
            CanvasOverlay[Tag canvas: Bounding Boxes + Nomes IA 60 FPS]
        end
    end

    Cam0 -->|RTSP Passthrough| go2rtc
    Cam1 -->|RTSP Passthrough| go2rtc
    Cam2 -->|RTSP Passthrough| go2rtc

    Cam0 -->|RTSP Leitura| IA_Engine
    Cam1 -->|RTSP Leitura| IA_Engine
    Cam2 -->|RTSP Leitura| IA_Engine

    go2rtc -->|WebRTC WHEP / RTP UDP| VideoTag
    IA_Engine -->|Coordenadas JSON| WS_Meta
    WS_Meta -->|Coordenadas JSON| CanvasOverlay
```

---

## 3. Especificação Técnica dos Módulos

### 3.1 Gateway WebRTC: `go2rtc`
- Utiliza a ferramenta de código aberto `go2rtc` (binário independente em Go, padrão ouro em sistemas como Frigate e Home Assistant).
- Arquivo de configuração: `go2rtc.yaml` na raiz do projeto Cerberus.
- As câmeras são expostas como streams WebRTC via protocolo padrão **WHEP** (*WebRTC HTTP Egress Protocol*):
  - `POST http://192.168.68.72:1984/api/whep?src=cam-0`
  - `POST http://192.168.68.72:1984/api/whep?src=cam-1`
  - `POST http://192.168.68.72:1984/api/whep?src=cam-2`
- Gerenciamento: pode rodar como serviço systemd dedicado (`go2rtc.service`) ou iniciado como subprocesso pelo Cerberus.

### 3.2 Canal de Metadados: `/ws/stream-meta/{camera_id}`
- Endpoint WebSocket dedicado e extremamente leve no FastAPI (`app/api/websocket.py`).
- Publica atualizações de rastros a cada ciclo de IA (ou quando houver alteração):
  ```json
  {
    "camera_id": "cam-0",
    "timestamp": 1729532400.12,
    "source_res": [2304, 1296],
    "detections": [
      {
        "tracking_id": 12,
        "bbox": [540, 210, 720, 680],
        "name": "Karan",
        "entity_type": "person",
        "class_label": "Pessoa",
        "is_known": true,
        "confidence": 0.94,
        "is_stationary": false
      }
    ]
  }
  ```

### 3.3 Interface Web: Video Tag + Canvas Overlay
- **Estrutura HTML**:
  ```html
  <div class="camera-stream-wrapper">
    <video id="video-cam-0" class="stream-video" autoplay playsinline muted></video>
    <canvas id="canvas-cam-0" class="stream-overlay"></canvas>
  </div>
  ```
- **Estilização CSS**:
  - `stream-video`: `width: 100%; height: 100%; object-fit: cover; display: block;`
  - `stream-overlay`: `position: absolute; inset: 0; pointer-events: none; width: 100%; height: 100%; z-index: 5;`
- **Renderização JavaScript**:
  - `app.js` conecta à URL WHEP do `go2rtc` usando a API nativa `RTCPeerConnection` do navegador.
  - O canvas desenha as bounding boxes com escala proporcional à resolução do vídeo.
  - Efeito suave de transição: caixas mantêm persistência entre inferências com interpolação de movimento.

### 3.4 Fallback Gracioso
- Se o `go2rtc` estiver indisponível ou a negociação WebRTC falhar após 3 segundos, o cliente ativa automaticamente o streaming via WebSocket JPEG atual (`/ws/stream/{cam_id}`), garantindo 100% de disponibilidade.

---

## 4. Plano de Implementação

1. **Configuração e Script de Instalação do go2rtc**:
   - Criar `go2rtc.yaml` apontando para as 3 câmeras RTSP.
   - Criar script `scripts/setup_go2rtc.sh` para download do binário oficial `go2rtc_linux_amd64` no Nobara Linux.
2. **Implementar Endpoint de Metadados**:
   - Adicionar `/ws/stream-meta/{camera_id}` em `app/api/websocket.py`.
3. **Atualizar Frontend**:
   - Ajustar `app/static/index.html` e `app/static/css/style.css` para a estrutura de camada dupla (`<video>` + `<canvas>`).
   - Implementar cliente WebRTC WHEP e renderizador de overlay em `app/static/js/app.js`.
4. **Verificação e Homologação**:
   - Testar conectividade WebRTC, sincronia visual das caixas e medição de FPS (alcançando 20–30 FPS).
