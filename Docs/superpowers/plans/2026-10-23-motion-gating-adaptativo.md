# Plano de Implementação: Motion Gating Adaptativo (Estilo Frigate)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) ou superpowers:executing-plans para implementar este plano tarefa por tarefa. As etapas utilizam a sintaxe de caixas de seleção (`- [ ]`) para rastreamento.

**Goal:** Implementar o filtro preliminar de movimento (Motion Gating com MOG2 do OpenCV) no Cerberus para pular inferências pesadas de IA (YOLOv8, FaceNet e Re-ID) quando a cena estiver estática, economizando ciclos de GPU, VRAM e energia sem perda de detecções.

**Architecture:** Módulo isolado `MotionDetector` baseado em OpenCV MOG2 rodando em resolução reduzida na CPU (~0.3ms). O `StreamManager` consulta o detector antes de submeter o frame ao YOLOv8. Se a cena estiver estática e não houver período de graça ativo nem rastros em acompanhamento (`has_active_tracks()`), o pipeline de IA é ignorado e entra em modo eco.

**Tech Stack:** Python 3.10+, OpenCV (`cv2.createBackgroundSubtractorMOG2`), Pydantic Settings, asyncio, pytest.

## Global Constraints

- Proibição absoluta de execução de scripts shell, runtimes Python e serviços no macOS (conforme `AGENTS.md`). Testes unitários sintéticos isolados e validação estática de tipos/código podem ser preparados.
- Todas as mensagens de commit devem ser escritas em português do Brasil.
- Comunicação sempre em português do Brasil.
- Preservação da integridade de comentários e docstrings existentes.

---

### Task 1: Configurações do Motion Gating no Pydantic

**Files:**
- Modify: `app/core/config.py:40-80`
- Test: `tests/test_config_motion.py`

**Interfaces:**
- Consumes: `pydantic_settings.BaseSettings`, `Field`
- Produces: `settings.motion_gating_enabled: bool`, `settings.motion_min_area: float`, `settings.motion_grace_period: float`

- [ ] **Step 1: Escrever teste unitário para novas configurações**

```python
import os
import pytest
from app.core.config import Settings

def test_motion_gating_default_settings():
    s = Settings()
    assert s.motion_gating_enabled is True
    assert s.motion_min_area == 0.005
    assert s.motion_grace_period == 3.0

def test_motion_gating_env_override(monkeypatch):
    monkeypatch.setenv("MOTION_GATING_ENABLED", "false")
    monkeypatch.setenv("MOTION_MIN_AREA", "0.02")
    monkeypatch.setenv("MOTION_GRACE_PERIOD", "5.5")
    s = Settings()
    assert s.motion_gating_enabled is False
    assert s.motion_min_area == 0.02
    assert s.motion_grace_period == 5.5
```

- [ ] **Step 2: Atualizar `app/core/config.py` com as novas opções**

Adicionar em `app/core/config.py`:
```python
    # Motion Gating Adaptativo (Estilo Frigate)
    motion_gating_enabled: bool = Field(default=True, alias="MOTION_GATING_ENABLED")
    motion_min_area: float = Field(default=0.005, alias="MOTION_MIN_AREA")
    motion_grace_period: float = Field(default=3.0, alias="MOTION_GRACE_PERIOD")
```

- [ ] **Step 3: Atualizar `.env.example`**

Adicionar as variáveis no arquivo `.env.example`:
```ini
# Motion Gating Adaptativo (Estilo Frigate - Economia de GPU)
MOTION_GATING_ENABLED=true
MOTION_MIN_AREA=0.005
MOTION_GRACE_PERIOD=3.0
```

- [ ] **Step 4: Commit**

```bash
git add app/core/config.py .env.example tests/test_config_motion.py
git commit -m "feat: adiciona configuracoes para motion gating adaptativo"
```

---

### Task 2: Implementar Módulo `MotionDetector` em `app/detector/motion.py`

**Files:**
- Create: `app/detector/motion.py`
- Test: `tests/test_motion_detector.py`

**Interfaces:**
- Consumes: `cv2`, `numpy`, `settings`
- Produces: `MotionDetector(min_area_ratio: float = 0.005, history: int = 500, var_threshold: int = 25)`
  - Método: `detect(frame: np.ndarray) -> Tuple[bool, float]`
  - Método: `reset() -> None`

- [ ] **Step 1: Escrever teste unitário para o `MotionDetector`**

```python
import numpy as np
import pytest
from app.detector.motion import MotionDetector

def test_motion_detector_static_frames():
    detector = MotionDetector(min_area_ratio=0.01)
    # Gera frames idênticos (fundo estático)
    static_frame = np.full((720, 1280, 3), 128, dtype=np.uint8)
    
    # Treina o subtractor com alguns frames estáticos
    for _ in range(5):
        has_motion, ratio = detector.detect(static_frame)
    
    # No fundo estático, não deve haver movimento significativo
    has_motion, ratio = detector.detect(static_frame)
    assert has_motion is False
    assert ratio < 0.01

def test_motion_detector_with_movement():
    detector = MotionDetector(min_area_ratio=0.01)
    base_frame = np.full((720, 1280, 3), 50, dtype=np.uint8)
    
    # Inicializa fundo
    for _ in range(10):
        detector.detect(base_frame)
        
    # Insere uma alteração visual ampla (quadrado branco grande)
    moved_frame = base_frame.copy()
    moved_frame[200:500, 400:800] = 255
    has_motion, ratio = detector.detect(moved_frame)
    assert has_motion is True
    assert ratio >= 0.01
```

- [ ] **Step 2: Implementar classe `MotionDetector` em `app/detector/motion.py`**

```python
from typing import Tuple
import cv2
import numpy as np
from app.core.logging import logger

class MotionDetector:
    """
    Detector de movimento ultraleve baseado no algoritmo MOG2 do OpenCV.
    Projetado para atuar como filtro pré-IA (Motion Gating), descartando
    frames estáticos antes do processamento pesado no YOLOv8/TensorRT.
    """
    def __init__(
        self,
        min_area_ratio: float = 0.005,
        history: int = 500,
        var_threshold: int = 25,
        target_width: int = 320
    ):
        self.min_area_ratio = min_area_ratio
        self.target_width = target_width
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )
        self.kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self._warmup_frames = 5
        self._frame_count = 0

    def detect(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Analisa o frame e retorna se houve variação significativa de pixels.
        Retorna (has_motion: bool, motion_ratio: float).
        """
        if frame is None or frame.size == 0:
            return False, 0.0

        h, w = frame.shape[:2]
        # Redução proporcional ultraleve para target_width (ex: 320x180)
        scale = self.target_width / float(w)
        target_h = int(h * scale)
        small_frame = cv2.resize(frame, (self.target_width, target_h), interpolation=cv2.INTER_NEAREST)
        
        # Converte para escala de cinza
        if len(small_frame.shape) == 3:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = small_frame

        # Suavização Gaussiana rápida para remover ruídos de alta frequência
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Subtração de fundo
        fg_mask = self.subtractor.apply(blurred)

        # Dilatação para conectar componentes conectados de movimento
        dilated = cv2.dilate(fg_mask, self.kernel, iterations=2)

        # Cálculo da proporção de pixels ativos
        total_pixels = float(target_h * self.target_width)
        active_pixels = cv2.countNonZero(dilated)
        motion_ratio = active_pixels / total_pixels if total_pixels > 0 else 0.0

        self._frame_count += 1
        # Durante o aquecimento do modelo de fundo, considera que há movimento para evitar perder eventos iniciais
        if self._frame_count <= self._warmup_frames:
            return True, motion_ratio

        has_motion = motion_ratio >= self.min_area_ratio
        return has_motion, motion_ratio

    def reset(self) -> None:
        """Reinicia o modelo de subtração de fundo."""
        self._frame_count = 0
```

- [ ] **Step 3: Commit**

```bash
git add app/detector/motion.py tests/test_motion_detector.py
git commit -m "feat: implementa modulo isolado de MotionDetector com MOG2"
```

---

### Task 3: Expor `has_active_tracks()` em `CameraTrackerManager`

**Files:**
- Modify: `app/detector/tracker.py:240-260`
- Test: `tests/test_tracker_active_tracks.py`

**Interfaces:**
- Produces: `CameraTrackerManager.has_active_tracks() -> bool`
- Regra: Retorna `True` se houver rastros ativos não estacionários ou com menos de 3.0s de inatividade.

- [ ] **Step 1: Escrever teste unitário para `has_active_tracks`**

```python
import pytest
from app.detector.tracker import CameraTrackerManager, TrackedEntity

def test_has_active_tracks_empty():
    manager = CameraTrackerManager("cam-test", "Camera Teste")
    assert manager.has_active_tracks() is False

def test_has_active_tracks_with_entity():
    manager = CameraTrackerManager("cam-test", "Camera Teste")
    entity = TrackedEntity(track_id=1, class_id=0, class_name="person", bbox=(10, 10, 50, 100), confidence=0.85)
    manager.active_tracks[1] = entity
    assert manager.has_active_tracks() is True
```

- [ ] **Step 2: Adicionar método `has_active_tracks` na classe `CameraTrackerManager`**

Em `app/detector/tracker.py`:
```python
    def has_active_tracks(self) -> bool:
        """
        Retorna True se houver entidades ativas sendo rastreadas na cena.
        Utilizado pelo Motion Gating para não suspender a IA enquanto houver
        pessoas, veículos ou animais no campo de visão.
        """
        return len(self.active_tracks) > 0
```

- [ ] **Step 3: Commit**

```bash
git add app/detector/tracker.py tests/test_tracker_active_tracks.py
git commit -m "feat: adiciona verificacao has_active_tracks no CameraTrackerManager"
```

---

### Task 4: Integrar Motion Gating no `StreamManager` (`manager.py`)

**Files:**
- Modify: `app/stream/manager.py:20-45, 274-335`
- Test: `tests/test_stream_manager_motion.py`

**Interfaces:**
- Consumes: `app.detector.motion.MotionDetector`, `settings.motion_gating_enabled`, `settings.motion_grace_period`
- Produces: Dicionário `self.motion_detectors[cam_id]`, controle de execução de inferência e telemetria atualizada (`motion_active`, `motion_gated`).

- [ ] **Step 1: Instanciar `MotionDetector` por câmera no `StreamManager`**

No método `__init__` e no loop de inicialização de câmeras:
```python
from app.detector.motion import MotionDetector

# No __init__:
self.motion_detectors: Dict[str, MotionDetector] = {}
self.last_motion_time: Dict[str, float] = {}

# Na inicialização de cada câmera:
self.motion_detectors[cam.id] = MotionDetector(min_area_ratio=settings.motion_min_area)
self.last_motion_time[cam.id] = time.time()
```

- [ ] **Step 2: Integrar o controle condicional no `_camera_inference_worker`**

No loop `_camera_inference_worker`:
```python
                # Motion Gating Adaptativo (Estilo Frigate)
                now = time.time()
                motion_detector = self.motion_detectors.get(cam.id)
                has_motion = True
                motion_ratio = 0.0

                if settings.motion_gating_enabled and motion_detector:
                    has_motion, motion_ratio = motion_detector.detect(raw_frame)
                    if has_motion:
                        self.last_motion_time[cam.id] = now

                # Avalia se a IA deve rodar:
                # 1. Movimento detectado no frame atual
                # 2. Período de graça ativo após último movimento
                # 3. Há rastros ativos sendo acompanhados pelo tracker
                time_since_motion = now - self.last_motion_time.get(cam.id, 0.0)
                in_grace_period = time_since_motion <= settings.motion_grace_period
                has_tracks = tracker.has_active_tracks() if tracker else False

                should_infer = (not settings.motion_gating_enabled) or has_motion or in_grace_period or has_tracks

                if not should_infer:
                    # Modo Eco: Atualiza telemetria informando que a inferência foi economizada
                    self.latest_telemetry[cam.id] = {
                        **(self.latest_telemetry.get(cam.id) or {}),
                        "motion_active": False,
                        "motion_gated": True,
                        "motion_ratio": round(motion_ratio, 4)
                    }
                    await asyncio.sleep(0.04)  # Repouso leve (~25 checks/s)
                    continue
```

- [ ] **Step 3: Atualizar telemetria quando a inferência rodar**

No bloco pós-inferência:
```python
                self.latest_telemetry[cam.id] = {
                    **(self.latest_telemetry.get(cam.id) or {}),
                    "motion_active": has_motion,
                    "motion_gated": False,
                    "motion_ratio": round(motion_ratio, 4)
                }
```

- [ ] **Step 4: Escrever teste de integração para validação de lógica**

Criar `tests/test_stream_manager_motion.py` simulando os estados de decisão (`should_infer`).

- [ ] **Step 5: Commit**

```bash
git add app/stream/manager.py tests/test_stream_manager_motion.py
git commit -m "feat: integra motion gating adaptativo no worker de inferencia do StreamManager"
```

---

### Task 5: Expor Métricas no Endpoint `/api/stats` e Documentação

**Files:**
- Modify: `app/api/routes.py` (ou onde a telemetria das câmeras é agregada)
- Modify: `README.md`
- Test: `tests/test_routes_telemetry.py`

**Interfaces:**
- Produces: Campos `"motion_active"` e `"motion_gated"` em `/api/cameras` ou `/api/stats`.
- Documentação no `README.md` explicando o funcionamento e como ajustar a sensibilidade.

- [ ] **Step 1: Verificar e expor os campos na resposta de telemetria das câmeras**
- [ ] **Step 2: Atualizar tabela de variáveis de ambiente no `README.md` com `MOTION_GATING_ENABLED`, `MOTION_MIN_AREA`, `MOTION_GRACE_PERIOD`**
- [ ] **Step 3: Commit**

```bash
git add app/api/routes.py README.md
git commit -m "docs: documenta motion gating adaptativo e expoe telemetria na api"
```
