"""
Cerberus AI — Cliente de Integração com Usina Solar (Huawei FusionSolar Kiosk)
Consulta a telemetria em tempo real (potência atual, rendimento do dia/mês/ano)
da usina solar a partir do microserviço FusionSolar Kiosk com cache inteligente e fallback resiliente.
"""

import os
import re
import json
import time
import urllib.request
import urllib.parse
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger("cerberus.fusionsolar")


def _clean_numeric(value_str: Optional[str]) -> Optional[float]:
    """Extrai valor numérico de strings como '3.80kW', '34.53kWh', '2.57(t)'."""
    if not value_str:
        return None
    match = re.search(r"[-+]?\d*\.?\d+", str(value_str))
    if match:
        try:
            return float(match.group(0))
        except ValueError:
            return None
    return None


def _format_address(raw_addr: Optional[str]) -> str:
    """Higieniza strings de endereço cruas como 'BrasilJi-ParanáJardim Capelasso...'."""
    if not raw_addr:
        return "Usina Solar"
    addr = str(raw_addr).strip()
    if "Ji-Paraná" in addr or "Ji-Parana" in addr:
        return "Ji-Paraná, RO"
    # Se já estiver limpo
    if len(addr) < 35:
        return addr
    return addr[:32] + "..."


class FusionSolarClient:
    """
    Cliente assíncrono para comunicação com o microsserviço FusionSolar Kiosk.
    Possui cache em memória, timeout curto e múltiplos níveis de fallback.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_fetch_ts: float = 0.0
        self._lock = asyncio.Lock()

    def _get_fallback_file_data(self) -> Optional[Dict[str, Any]]:
        """Tenta carregar o último snapshot salvo em arquivo local caso o serviço HTTP esteja offline."""
        candidate_paths = [
            Path("/Users/karan/Github/Fusionsolar_kiosk/dados_solar.json"),
            Path(settings.data_dir) / "dados_solar.json",
            Path("dados_solar.json"),
            Path("../Fusionsolar_kiosk/dados_solar.json")
        ]

        for p in candidate_paths:
            try:
                if p.is_file():
                    with open(p, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if data and isinstance(data, dict):
                            return data
            except Exception:
                continue
        return None

    def _sync_http_fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """Executa requisição HTTP síncrona com timeout estrito em threadpool."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Cerberus-Solar-Client/1.0",
                    "Accept": "application/json"
                }
            )
            with urllib.request.urlopen(req, timeout=3.5) as response:
                if response.status == 200:
                    body = response.read().decode("utf-8")
                    return json.loads(body)
        except Exception as e:
            logger.debug(f"[FusionSolar] Falha na requisição HTTP: {e}")
            return None
        return None

    def _parse_solar_payload(self, raw: Dict[str, Any], is_fallback: bool = False) -> Dict[str, Any]:
        """Normaliza as métricas da usina para o padrão do Cerberus."""
        pot_str = raw.get("potencia_realtime", "")
        rend_hoje_str = raw.get("rendimento_hoje", "")
        rend_mes_str = raw.get("rendimento_mes", "")
        ano_str = raw.get("geracao_ano", "")
        co2_str = raw.get("co2_evitado", "")
        addr_str = raw.get("endereco", "")

        pot_num = _clean_numeric(pot_str) or 0.0
        rend_hoje_num = _clean_numeric(rend_hoje_str) or 0.0
        rend_mes_num = _clean_numeric(rend_mes_str) or 0.0
        co2_num = _clean_numeric(co2_str) or 0.0

        # Formatações elegantes para visualização em tela
        pot_display = f"{pot_num:.2f} kW" if pot_num > 0 else (pot_str or "0.00 kW")
        rend_hoje_display = f"{rend_hoje_num:.1f} kWh" if rend_hoje_num > 0 else (rend_hoje_str or "0.0 kWh")

        return {
            "available": True,
            "potencia_realtime": pot_display,
            "potencia_kw": pot_num,
            "rendimento_hoje": rend_hoje_display,
            "rendimento_hoje_kwh": rend_hoje_num,
            "rendimento_mes": rend_mes_str or f"{rend_mes_num:.1f} kWh",
            "geracao_ano": ano_str or "--",
            "co2_evitado": co2_str or f"{co2_num:.1f} t",
            "endereco": _format_address(addr_str),
            "data_hora": raw.get("data_hora") or time.strftime("%d/%m/%Y %H:%M"),
            "curva_potencia": raw.get("curva_potencia") or [],
            "is_fallback": is_fallback,
            "updated_at": time.time()
        }

    async def get_kiosk_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Retorna a telemetria da usina solar.
        Usa cache de memória (TTL definido em settings.fusionsolar_cache_seconds).
        Em caso de indisponibilidade da API, usa snapshot em arquivo ou último cache.
        """
        if not settings.fusionsolar_enabled:
            return {"available": False, "reason": "disabled"}

        now = time.time()
        ttl = max(30, settings.fusionsolar_cache_seconds)

        # 1. Verifica cache em memória
        if not force and self._cache and (now - self._last_fetch_ts < ttl):
            return self._cache

        async with self._lock:
            # Revalida dentro do lock para evitar thundering herd
            if not force and self._cache and (now - self._last_fetch_ts < ttl):
                return self._cache

            target_api_url = settings.fusionsolar_api_url.rstrip("/")
            kiosk_param = urllib.parse.quote(settings.fusionsolar_kiosk_url, safe="")
            endpoint = f"{target_api_url}/api/kiosk?url={kiosk_param}"

            # 2. Tenta consultar a API Rust do microserviço
            raw_data = await asyncio.to_thread(self._sync_http_fetch, endpoint)

            if raw_data and isinstance(raw_data, dict) and "potencia_realtime" in raw_data:
                parsed = self._parse_solar_payload(raw_data, is_fallback=False)
                self._cache = parsed
                self._last_fetch_ts = now
                return parsed

            # 3. Fallback: Arquivo local dados_solar.json
            fallback_data = await asyncio.to_thread(self._get_fallback_file_data)
            if fallback_data:
                parsed = self._parse_solar_payload(fallback_data, is_fallback=True)
                self._cache = parsed
                self._last_fetch_ts = now
                return parsed

            # 4. Fallback: Cache expirado anterior
            if self._cache:
                stale = dict(self._cache)
                stale["is_stale"] = True
                return stale

            # 5. Se nada estiver disponível
            return {
                "available": False,
                "reason": "kiosk_unreachable",
                "potencia_realtime": "0.00 kW",
                "rendimento_hoje": "0.0 kWh"
            }


fusionsolar_client = FusionSolarClient()
