import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CameraConfig(BaseModel):
    id: str
    name: str
    url: str
    active: bool = True
    confidence_threshold: float = 0.65
    priority: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Servidor
    host: str = Field(default="0.0.0.0", alias="HOST")
    port: int = Field(default=8095, alias="PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Hardware & Aceleração
    # Opções: auto, cuda:0, mps, cpu
    device_preference: str = Field(default="auto", alias="DEVICE")
    cuda_visible_devices: Optional[str] = Field(default="0", alias="CUDA_VISIBLE_DEVICES")

    # Arquivos e Diretórios
    model_path: str = Field(default="models/yolo11n.pt", alias="MODEL_PATH")
    config_path: str = Field(default="config/cameras.json", alias="CONFIG_PATH")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    events_dir: str = Field(default="data/events", alias="EVENTS_DIR")
    db_path: str = Field(default="data/events.db", alias="DB_PATH")

    # Parâmetros de Inferência e Anti-Duplicação
    confidence_threshold: float = Field(default=0.40, alias="CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(default=0.45, alias="IOU_THRESHOLD")
    alert_cooldown_seconds: float = Field(default=15.0, alias="ALERT_COOLDOWN_SECONDS")
    min_consecutive_frames: int = Field(default=2, alias="MIN_CONSECUTIVE_FRAMES")
    vehicle_similarity_threshold: float = Field(default=0.82, alias="VEHICLE_SIMILARITY_THRESHOLD")
    pet_similarity_threshold: float = Field(default=0.78, alias="PET_SIMILARITY_THRESHOLD")

    # Motion Gating Adaptativo (Estilo Frigate - Economia de GPU)
    motion_gating_enabled: bool = Field(default=True, alias="MOTION_GATING_ENABLED")
    motion_min_area: float = Field(default=0.005, alias="MOTION_MIN_AREA")
    motion_grace_period: float = Field(default=3.0, alias="MOTION_GRACE_PERIOD")

    # Parâmetros SOTA de Reconhecimento Facial (InsightFace ArcFace / SCRFD)
    face_model_name: str = Field(default="buffalo_l", alias="FACE_MODEL_NAME")
    face_det_thresh: float = Field(default=0.22, alias="FACE_DET_THRESH")
    face_confidence_threshold: float = Field(default=0.38, alias="FACE_CONFIDENCE_THRESHOLD")
    face_instant_threshold: float = Field(default=0.48, alias="FACE_INSTANT_THRESHOLD")
    face_min_size: int = Field(default=18, alias="FACE_MIN_SIZE")
    face_quality_min_blur: float = Field(default=4.5, alias="FACE_QUALITY_MIN_BLUR")
    face_temporal_votes: int = Field(default=2, alias="FACE_TEMPORAL_VOTES")
    face_augmentation_enabled: bool = Field(default=True, alias="FACE_AUGMENTATION_ENABLED")

    # Módulo de Restauração e Super-Resolução Facial SOTA
    face_shadow_lifting_enabled: bool = Field(default=True, alias="FACE_SHADOW_LIFTING_ENABLED")
    face_shadow_clip_limit: float = Field(default=3.0, alias="FACE_SHADOW_CLIP_LIMIT")
    face_shadow_gamma: float = Field(default=0.60, alias="FACE_SHADOW_GAMMA")
    face_super_res_enabled: bool = Field(default=True, alias="FACE_SUPER_RES_ENABLED")
    face_super_res_threshold: int = Field(default=75, alias="FACE_SUPER_RES_THRESHOLD")

    # Aceleração Máxima TensorRT (NVIDIA CUDA)
    use_tensorrt: bool = Field(default=True, alias="USE_TENSORRT")
    engine_path: str = Field(default="models/yolo11n.engine", alias="ENGINE_PATH")
    facenet_engine_path: str = Field(default="models/facenet.engine", alias="FACENET_ENGINE_PATH")

    # Decodificação de Vídeo em Hardware na GPU (NVIDIA NVDEC)
    use_nvdec: bool = Field(default=True, alias="USE_NVDEC")
    nvdec_device: int = Field(default=0, alias="NVDEC_DEVICE")

    # Resolução do Stream Web (Server-Side Streaming)
    # 1280 = 720p HD cristalino calibrado para mosaico 2x2 sem saturação de banda
    stream_max_width: int = Field(default=1280, alias="STREAM_MAX_WIDTH")
    stream_quality: int = Field(default=80, alias="STREAM_QUALITY")

    # Módulo de Notificação & Bot do Telegram (Bot 1 - Feed Geral / Silencioso)
    telegram_enabled: bool = Field(default=False, alias="TELEGRAM_ENABLED")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    telegram_notify_person: bool = Field(default=True, alias="TELEGRAM_NOTIFY_PERSON")
    telegram_notify_vehicle: bool = Field(default=True, alias="TELEGRAM_NOTIFY_VEHICLE")
    telegram_notify_pet: bool = Field(default=False, alias="TELEGRAM_NOTIFY_PET")
    telegram_send_both_photos: bool = Field(default=True, alias="TELEGRAM_SEND_BOTH_PHOTOS")

    # Módulo de Inteligência Artificial Contextual (OpenRouter) & Bot Crítico (Bot 2 - Alertas Prioritários)
    openrouter_api_key: str = Field(default="", alias="OPENROUTER_API_KEY")
    openrouter_model: str = Field(default="google/gemini-2.5-flash", alias="OPENROUTER_MODEL")
    openrouter_timeout: float = Field(default=12.0, alias="OPENROUTER_TIMEOUT")

    # Bot 2 - Canal de Alertas Críticos (Notificação com Som / Prioridade Máxima)
    telegram_critical_enabled: bool = Field(default=False, alias="TELEGRAM_CRITICAL_ENABLED")
    telegram_critical_bot_token: str = Field(default="", alias="TELEGRAM_CRITICAL_BOT_TOKEN")
    telegram_critical_chat_id: str = Field(default="", alias="TELEGRAM_CRITICAL_CHAT_ID")

    # Módulo Edge VLM Local (Moondream2 / SmolVLM - Visão Computacional Generativa 100% Offline na GPU)
    enable_edge_vlm: bool = Field(default=True, alias="ENABLE_EDGE_VLM")
    edge_vlm_model: str = Field(default="vikhyatk/moondream2", alias="EDGE_VLM_MODEL")
    edge_vlm_revision: str = Field(default="", alias="EDGE_VLM_REVISION")

    # Módulo de Integração com Usina Solar (Huawei FusionSolar Kiosk)
    fusionsolar_enabled: bool = Field(default=True, alias="FUSIONSOLAR_ENABLED")
    fusionsolar_token: str = Field(default="EM6wrnyAhBtOFvD7rdNu045xGpwb1bGD", alias="FUSIONSOLAR_TOKEN")
    fusionsolar_kiosk_url: str = Field(
        default="https://intl.fusionsolar.huawei.com/pvmswebsite/nologin/assets/build/cloud.html#/kiosk?kk=EM6wrnyAhBtOFvD7rdNu045xGpwb1bGD",
        alias="FUSIONSOLAR_KIOSK_URL"
    )
    fusionsolar_cache_seconds: int = Field(default=90, alias="FUSIONSOLAR_CACHE_SECONDS")

    # Coordenadas Geográficas de Ji-Paraná, RO (para Previsão do Tempo e Irradiação Solar)
    weather_lat: float = Field(default=-10.8847, alias="WEATHER_LAT")
    weather_lon: float = Field(default=-61.9514, alias="WEATHER_LON")
    weather_city_name: str = Field(default="Ji-Paraná, RO", alias="WEATHER_CITY_NAME")

    def resolve_device(self) -> str:
        """
        Resolve dinamicamente o melhor dispositivo de inferência disponível,
        garantindo compatibilidade com o servidor Nobara Linux (CUDA) e o
        ambiente de desenvolvimento no Mac M4 (MPS/CPU).
        """
        try:
            import torch
        except ImportError:
            return "cpu"

        pref = (self.device_preference or "auto").strip().lower()

        # Se o usuário pediu explicitamente CUDA ou auto
        if pref.startswith("cuda") or pref == "auto":
            if torch.cuda.is_available():
                # Define variável de ambiente para manter compatibilidade em setups Dual-GPU
                if self.cuda_visible_devices is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
                    os.environ["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)
                return "cuda:0" if pref == "auto" else pref
            elif pref.startswith("cuda"):
                # Aviso de fallback silencioso caso o desenvolvedor rode no Mac M4
                pass

        # Se estiver no macOS Apple Silicon (M1/M2/M3/M4)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    def ensure_directories(self) -> None:
        """Garante a existência de pastas necessárias para dados e modelos."""
        Path(self.data_dir).mkdir(parents=True, exist_ok=True)
        Path(self.events_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)


settings = Settings()
