# Especificação de Design: Motion Gating Adaptativo (Estilo Frigate)

**Data:** 2026-10-23  
**Status:** Aprovado  
**Objetivo:** Reduzir drasticamente o consumo ocioso de VRAM, temperatura e ciclos de inferência da GPU NVIDIA GTX 1660 no Cerberus, pulando inferências pesadas de IA (YOLOv8, FaceNet e Re-ID) quando a cena estiver estática, sem perda de acurácia ou reatividade.

---

## 1. Visão Geral e Motivação

Atualmente, o worker de inferência (`_camera_inference_worker`) submete frames para detecção no YOLOv8 e extratores biométricos continuamente, mesmo quando a câmera aponta para um cenário completamente estático (ex.: portão ou garagem vazios de madrugada).

Embora a aceleração TensorRT FP16 seja rápida (~3 a 4ms), executar continuamente 20 a 30 FPS de inferência em múltiplas câmeras causa:
1. Consumo contínuo de energia e aquecimento desnecessário da GPU.
2. Competição desnecessária de barramento PCI-e e memória VRAM.
3. Menos margem de throughput para novas câmeras ou análises aprofundadas.

Inspirado na arquitetura do **Frigate NVR**, a técnica de **Motion Gating Adaptativo** filtra a entrada da IA: apenas frames que apresentem movimentação perceptível disparam o pipeline pesado.

---

## 2. Arquitetura do Sistema

```text
[ RTSP Frame ] ──> [ MotionDetector (Downscale 320x180 + MOG2) ]
                                 │
                 ┌───────────────┴───────────────┐
                 ▼                               ▼
       [ Movimento Detectado ]         [ Cena Estática ]
                 │                               │
                 │                     ┌─────────┴─────────┐
                 │                     ▼                   ▼
                 │            [ Período de Graça?   [ Rastreamento Ativo? ]
                 │              (<= 3 segundos) ]   (has_active_tracks)
                 │                     │                   │
                 ▼                     ▼                   ▼
        ┌────────────────────────────────────────────────────────┐
        │  SIM: Dispara Pipeline de IA (YOLOv8 + Biometria)      │
        └────────────────────────────────────────────────────────┘
                                       │ NÃO
                                       ▼
        ┌────────────────────────────────────────────────────────┐
        │  PULA inferência, dorme levemente (eco-mode)          │
        │  e atualiza telemetria (motion_gated = True)           │
        └────────────────────────────────────────────────────────┘
```

---

## 3. Especificação dos Componentes

### 3.1. Módulo `app/detector/motion.py` (`MotionDetector`)
Cada câmera possui sua própria instância isolada de `MotionDetector` para manter o modelo de subtração de fundo específico para seu enquadramento e iluminação.

* **Downscaling rápido:** O frame de entrada é reduzido para resolução baixa (ex: 320x180) e convertido para tons de cinza (`cv2.COLOR_BGR2GRAY`), custando menos de 0.3ms na CPU.
* **Subtração de Fundo (MOG2):**
  * `cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)`
  * Aprende variações graduais de luz do dia e ignora oscilações suaves.
* **Morfologia:** Operações de `cv2.morphologyEx` ou `cv2.dilate` com kernel 3x3 para eliminar ruídos de sensores.
* **Métricas:** Calcula o somatório dos pixels ativos em relação à área total do frame (`motion_ratio`).
* **Saída:** `(has_motion: bool, motion_ratio: float)`.

### 3.2. Configurações (`app/core/config.py`)
Novas opções com suporte a variáveis de ambiente e `.env`:
* `MOTION_GATING_ENABLED: bool = True` (flag geral de ativação).
* `MOTION_MIN_AREA: float = 0.005` (limiar de 0.5% da área da imagem com movimento).
* `MOTION_GRACE_PERIOD: float = 3.0` (tempo de tolerância em segundos após o último movimento para manter a IA acompanhando a cena).

### 3.3. Integração com Rastreador (`app/detector/tracker.py`)
* O `CameraTrackerManager` expõe o método `has_active_tracks() -> bool` para verificar se há algum rastro de pessoa/veículo/animal em acompanhamento ativo na cena.
* Se houver rastros ativos, a inferência NÃO é interrompida, mesmo que o indivíduo fique parado temporariamente (ex.: pessoa parada conversando no portão).

### 3.4. Orquestrador (`app/stream/manager.py`)
* No `_camera_inference_worker`:
  1. Avalia o `MotionDetector`.
  2. Se houver movimento, atualiza `last_motion_time = time.time()`.
  3. Se não houver movimento mas `now - last_motion_time <= MOTION_GRACE_PERIOD` ou `tracker.has_active_tracks()` for verdadeiro: prossegue com a inferência.
  4. Caso contrário:
     * Pula a execução do YOLO.
     * Atualiza telemetria (`is_gated = True`, `has_motion = False`).
     * Realiza `await asyncio.sleep(0.04)` para não consumir CPU desnecessariamente.

---

## 4. Requisitos Não Funcionais e Segurança
* **Compatibilidade com Hardware:** O pré-processamento de movimento é executado em CPU via OpenCV sem alocação de memória na GPU.
* **Zero Falha em Detecções:** A combinação do algoritmo MOG2 com o período de graça de 3.0s e verificação de rastros ativos assegura que nenhuma pessoa ou veículo deixe de ser rastreado.
* **Desenvolvimento e Deploy:** Compatível com execução local em macOS e produção no Nobara Linux 44 (respeitando a regra de não execução no Mac).
