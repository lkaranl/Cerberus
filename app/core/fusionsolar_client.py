"""
Cerberus AI — Cliente de Integração Direta com Usina Solar (Huawei FusionSolar Kiosk)
Extrai telemetria em tempo real (potência atual em kW, geração acumulada em kWh, etc.)
diretamente da URL pública do quiosque da Huawei (via endpoint REST nativo),
sem depender de nenhum microserviço intermediário ou instância de navegador headless.
"""

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


def _format_address(raw_addr: Optional[str]) -> str:
    """Higieniza strings de endereço cruas como 'BrasilJi-ParanáJardim Capelasso...'."""
    if not raw_addr:
        return "Usina Solar"
    addr = str(raw_addr).strip()
    if "Ji-Paraná" in addr or "Ji-Parana" in addr:
        return "Ji-Paraná, RO"
    if len(addr) < 32:
        return addr
    return addr[:30] + "..."


def _extract_kiosk_token(kiosk_url: str) -> Optional[str]:
    """Extrai o token 'kk' a partir de qualquer formato de URL pública do FusionSolar."""
    if not kiosk_url:
        return None
    match = re.search(r"kk=([a-zA-Z0-9_\-]+)", kiosk_url)
    return match.group(1) if match else None


def _build_huawei_rest_url(kiosk_url: str) -> Optional[str]:
    """
    Constrói a URL do endpoint REST JSON nativo da Huawei a partir da URL do quiosque.
    Ex:
    Entrada: https://intl.fusionsolar.huawei.com/pvmswebsite/nologin/assets/build/cloud.html#/kiosk?kk=TOKEN
    Saída:   https://intl.fusionsolar.huawei.com/rest/pvms/web/kiosk/v1/station-kiosk-file?kk=TOKEN
    """
    token = _extract_kiosk_token(kiosk_url)
    if not token:
        return None

    try:
        parsed = urllib.parse.urlparse(kiosk_url)
        scheme = parsed.scheme or "https"
        netloc = parsed.netloc or "intl.fusionsolar.huawei.com"
        return f"{scheme}://{netloc}/rest/pvms/web/kiosk/v1/station-kiosk-file?kk={token}"
    except Exception as e:
        logger.debug(f"[FusionSolar] Erro ao parsear URL do quiosque: {e}")
        return None


class FusionSolarClient:
    """
    Cliente assíncrono para extração direta de telemetria da Huawei FusionSolar.
    Executa requisição HTTP direta ao endpoint JSON oficial da Huawei com cache em memória.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_fetch_ts: float = 0.0
        self._lock = asyncio.Lock()

    def _sync_http_fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """Executa requisição HTTP síncrona com timeout estrito em threadpool."""
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)",
                    "Accept": "application/json, text/plain, */*",
                    "Referer": settings.fusionsolar_kiosk_url
                }
            )
            with urllib.request.urlopen(req, timeout=4.0) as response:
                if response.status == 200:
                    body = response.read().decode("utf-8")
                    return json.loads(body)
        except Exception as e:
            logger.debug(f"[FusionSolar] Falha na consulta REST à Huawei: {e}")
            return None
        return None

    def _get_fallback_file_data(self) -> Optional[Dict[str, Any]]:
        """Snapshot local caso a conexão externa à Huawei falhe temporariamente."""
        candidate_paths = [
            Path("/Users/karan/Github/Fusionsolar_kiosk/dados_solar.json"),
            Path(settings.data_dir) / "dados_solar.json",
            Path("dados_solar.json"),
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

    def _parse_huawei_payload(self, raw_resp: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Decodifica a estrutura retornada pelo endpoint REST da Huawei:
        {"success": true, "data": "{\"realKpi\": ...}"}
        """
        try:
            raw_data = raw_resp.get("data")
            if not raw_data:
                return None

            # O payload interno 'data' pode vir como string JSON codificada
            if isinstance(raw_data, str):
                station_data = json.loads(raw_data)
            elif isinstance(raw_data, dict):
                station_data = raw_data
            else:
                return None

            real_kpi = station_data.get("realKpi", {})
            station = station_data.get("stationOverview", {})
            social = station_data.get("socialContribution", {})
            power_curve = station_data.get("powerCurve", {})

            # Métricas numéricas primárias
            power_num = float(real_kpi.get("realTimePower") or 0.0)
            daily_num = float(real_kpi.get("dailyEnergy") or 0.0)
            month_num = float(real_kpi.get("monthEnergy") or 0.0)
            year_num = float(real_kpi.get("yearEnergy") or 0.0)
            cumulative_num = float(real_kpi.get("cumulativeEnergy") or 0.0)

            # Impacto social / ecológico
            co2_kg = float(social.get("co2Reduction") or 0.0)
            co2_tons = co2_kg / 1000.0 if co2_kg > 100 else co2_kg
            trees = int(social.get("equivalentTreePlanting") or 0)

            station_name = station.get("stationName") or "Usina Solar"
            addr = _format_address(station.get("plantAddress"))

            # Formatações elegantes para visualização no dashboard e TV
            pot_display = f"{power_num:.2f} kW"
            daily_display = f"{daily_num:.1f} kWh"
            month_display = f"{month_num:.1f} kWh" if month_num < 1000 else f"{(month_num/1000.0):.2f} MWh"
            year_display = f"{year_num:.1f} kWh" if year_num < 1000 else f"{(year_num/1000.0):.2f} MWh"

            # Pontos horários da curva para gráfico/inspeção
            x_axis = power_curve.get("xAxis") or []
            active_power = power_curve.get("activePower") or []
            curva = []
            for h, p in zip(x_axis, active_power):
                if p and p != "-":
                    curva.append({"hora": h, "potencia": f"{p} kW"})

            return {
                "available": True,
                "station_name": station_name,
                "potencia_realtime": pot_display,
                "potencia_kw": power_num,
                "rendimento_hoje": daily_display,
                "rendimento_hoje_kwh": daily_num,
                "rendimento_mes": month_display,
                "geracao_ano": year_display,
                "eletricidade_acumulada": f"{(cumulative_num/1000.0):.2f} MWh" if cumulative_num >= 1000 else f"{cumulative_num:.1f} kWh",
                "co2_evitado": f"{co2_tons:.2f} t",
                "arvores_plantadas": str(trees),
                "endereco": addr,
                "data_hora": time.strftime("%d/%m/%Y %H:%M"),
                "curva_potencia": curva,
                "updated_at": time.time()
            }
        except Exception as e:
            logger.warning(f"[FusionSolar] Erro ao decodificar payload REST da Huawei: {e}")
            return None

    async def get_kiosk_data(self, force: bool = False) -> Dict[str, Any]:
        """
        Retorna a telemetria da usina solar diretamente da URL pública do quiosque da Huawei.
        Usa cache de memória (TTL definido em settings.fusionsolar_cache_seconds).
        """
        if not settings.fusionsolar_enabled:
            return {"available": False, "reason": "disabled"}

        kiosk_url = settings.fusionsolar_kiosk_url
        if not kiosk_url:
            return {"available": False, "reason": "missing_kiosk_url"}

        now = time.time()
        ttl = max(30, settings.fusionsolar_cache_seconds)

        # 1. Cache em memória válido
        if not force and self._cache and (now - self._last_fetch_ts < ttl):
            return self._cache

        async with self._lock:
            if not force and self._cache and (now - self._last_fetch_ts < ttl):
                return self._cache

            # 2. Constrói a URL REST direta da Huawei
            rest_url = _build_huawei_rest_url(kiosk_url)
            if rest_url:
                raw_json = await asyncio.to_thread(self._sync_http_fetch, rest_url)
                if raw_json and raw_json.get("success"):
                    parsed = self._parse_huawei_payload(raw_json)
                    if parsed:
                        self._cache = parsed
                        self._last_fetch_ts = now
                        return parsed

            # 3. Fallback: Arquivo local se existir
            fallback_data = await asyncio.to_thread(self._get_fallback_file_data)
            if fallback_data:
                pot_str = fallback_data.get("potencia_realtime", "0.00kW")
                rend_str = fallback_data.get("rendimento_hoje", "0.0kWh")
                pot_num = float(re.search(r"\d+\.?\d*", pot_str).group(0)) if re.search(r"\d+\.?\d*", pot_str) else 0.0
                rend_num = float(re.search(r"\d+\.?\d*", rend_str).group(0)) if re.search(r"\d+\.?\d*", rend_str) else 0.0

                parsed = {
                    "available": True,
                    "station_name": "Usina Solar",
                    "potencia_realtime": f"{pot_num:.2f} kW",
                    "potencia_kw": pot_num,
                    "rendimento_hoje": f"{rend_num:.1f} kWh",
                    "rendimento_hoje_kwh": rend_num,
                    "rendimento_mes": fallback_data.get("rendimento_mes", "--"),
                    "geracao_ano": fallback_data.get("geracao_ano", "--"),
                    "co2_evitado": fallback_data.get("co2_evitado", "--"),
                    "endereco": _format_address(fallback_data.get("endereco")),
                    "data_hora": time.strftime("%d/%m/%Y %H:%M"),
                    "curva_potencia": [],
                    "is_fallback": True,
                    "updated_at": time.time()
                }
                self._cache = parsed
                self._last_fetch_ts = now
                return parsed

            # 4. Fallback: Cache anterior expirado
            if self._cache:
                stale = dict(self._cache)
                stale["is_stale"] = True
                return stale

            return {
                "available": False,
                "reason": "huawei_unreachable",
                "potencia_realtime": "0.00 kW",
                "rendimento_hoje": "0.0 kWh"
            }


fusionsolar_client = FusionSolarClient()
