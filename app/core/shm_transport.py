"""
Cerberus AI — Transporte Zero-Copy de Imagens via Memória Compartilhada POSIX (/dev/shm)
Permite a troca instantânea de matrizes NumPy entre o processo principal (NVIDIA)
e o worker especialista (AMD) sem overhead de compressão JPEG ou codificação Base64.
"""

import os
import uuid
import logging
from typing import Tuple, List, Optional
from multiprocessing import shared_memory
import numpy as np

logger = logging.getLogger("cerberus.shm")

def write_numpy_to_shm(arr: np.ndarray, prefix: str = "cerb") -> Tuple[str, List[int], str, shared_memory.SharedMemory]:
    """
    Grava um array NumPy contíguo em um novo bloco de memória compartilhada (/dev/shm).
    Retorna: (nome_do_bloco, shape, dtype_str, objeto_shm).
    O chamador deve manter objeto_shm aberto até o receptor consumir, ou chamar shm.close().
    """
    if not arr.flags["C_CONTIGUOUS"]:
        arr = np.ascontiguousarray(arr)

    unique_id = uuid.uuid4().hex[:12]
    shm_name = f"{prefix}_{unique_id}"
    size_bytes = arr.nbytes

    shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size_bytes)
    # Cópia direta e instantânea em memória RAM (Zero-Copy virtual)
    shm_arr = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    shm_arr[:] = arr[:]

    return shm_name, list(arr.shape), str(arr.dtype), shm

def read_numpy_from_shm(shm_name: str, shape: List[int], dtype_str: str, unlink: bool = True) -> Optional[np.ndarray]:
    """
    Mapeia e lê um array NumPy a partir de um bloco em /dev/shm existente.
    Se unlink=True, remove o arquivo de memória compartilhada do sistema após a leitura.
    """
    try:
        shm = shared_memory.SharedMemory(name=shm_name, create=False)
        # Cria visualização sobre o buffer
        shm_arr = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=shm.buf)
        data_copy = shm_arr.copy()
        shm.close()
        if unlink:
            try:
                shm.unlink()
            except Exception:
                pass
        return data_copy
    except Exception as e:
        logger.debug(f"Falha ao ler bloco de memória compartilhada {shm_name}: {e}")
        return None

def cleanup_stale_shm(prefix: str = "cerb_"):
    """Remove blocos temporários de memória compartilhada órfãos em /dev/shm."""
    try:
        shm_dir = "/dev/shm"
        if os.path.exists(shm_dir):
            for fname in os.listdir(shm_dir):
                if fname.startswith(prefix):
                    try:
                        os.unlink(os.path.join(shm_dir, fname))
                    except Exception:
                        pass
    except Exception:
        pass
