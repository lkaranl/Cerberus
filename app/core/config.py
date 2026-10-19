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
    model_path: str = Field(default="models/yolov8n.pt", alias="MODEL_PATH")
    config_path: str = Field(default="config/cameras.json", alias="CONFIG_PATH")
    data_dir: str = Field(default="data", alias="DATA_DIR")
    events_dir: str = Field(default="data/events", alias="EVENTS_DIR")
    db_path: str = Field(default="data/events.db", alias="DB_PATH")

    # Parâmetros de Inferência e Anti-Duplicação
    confidence_threshold: float = Field(default=0.65, alias="CONFIDENCE_THRESHOLD")
    iou_threshold: float = Field(default=0.45, alias="IOU_THRESHOLD")
    alert_cooldown_seconds: float = Field(default=45.0, alias="ALERT_COOLDOWN_SECONDS")
    min_consecutive_frames: int = Field(default=6, alias="MIN_CONSECUTIVE_FRAMES")
    vehicle_similarity_threshold: float = Field(default=0.82, alias="VEHICLE_SIMILARITY_THRESHOLD")
    pet_similarity_threshold: float = Field(default=0.80, alias="PET_SIMILARITY_THRESHOLD")

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
