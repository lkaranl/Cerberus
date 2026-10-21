"""
Cerberus AI — Transporte Zero-Copy de Imagens via Memória Compartilhada POSIX (/dev/shm)
Permite a troca instantânea de matrizes NumPy entre o processo principal (NVIDIA)
e o worker especialista (AMD) sem overhead de compressão JPEG, sem Base64 e sem
avisos de resource_tracker do Python.
"""

import os
import uuid
import logging
from typing import Tuple, List, Optional
import numpy as np

logger = logging.getLogger("cerberus.shm")
SHM_DIR = "/dev/shm" if os.path.exists("/dev/shm") else "/tmp"

def write_numpy_to_shm(arr: np.ndarray, prefix: str = "cerb") -> Tuple[str, List[int], str, str]:
    """
    Grava um array NumPy contíguo em um arquivo raw diretamente na memória RAM física (/dev/shm).
    Retorna: (shm_name, shape, dtype_str, file_path).
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    unique_id = uuid.uuid4().hex[:12]
    shm_name = f"{prefix}_{unique_id}.raw"
    file_path = os.path.join(SHM_DIR, shm_name)

    try:
        with open(file_path, "wb") as f:
            f.write(arr.tobytes())
    except Exception as e:
        logger.debug(f"Falha ao gravar em {file_path}: {e}")

    return shm_name, list(arr.shape), str(arr.dtype), file_path

def read_numpy_from_shm(shm_name: str, shape: List[int], dtype_str: str, unlink: bool = True) -> Optional[np.ndarray]:
    """
    Lê a matriz NumPy diretamente da RAM física (/dev/shm) e remove o arquivo em seguida.
    """
    file_path = os.path.join(SHM_DIR, shm_name)
    try:
        if not os.path.exists(file_path):
            return None

        with open(file_path, "rb") as f:
            raw_bytes = f.read()

        arr = np.frombuffer(raw_bytes, dtype=np.dtype(dtype_str)).reshape(shape)
        data_copy = arr.copy()

        if unlink:
            try:
                os.unlink(file_path)
            except Exception:
                pass

        return data_copy
    except Exception as e:
        logger.debug(f"Falha ao ler {file_path} em /dev/shm: {e}")
        return None

def cleanup_stale_shm(prefix: str = "cerb_"):
    """Remove arquivos residuais em /dev/shm com o prefixo do Cerberus."""
    try:
        if os.path.exists(SHM_DIR):
            for fname in os.listdir(SHM_DIR):
                if fname.startswith(prefix) and fname.endswith(".raw"):
                    try:
                        os.unlink(os.path.join(SHM_DIR, fname))
                    except Exception:
                        pass
    except Exception:
        pass
