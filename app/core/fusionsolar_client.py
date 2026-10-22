"""
Cerberus AI — Cliente de Integração Direta com Usina Solar (Huawei FusionSolar Kiosk)
Extrai telemetria em tempo real (potência atual em kW, geração acumulada em kWh, etc.)
diretamente da URL pública do quiosque da Huawei (via endpoint REST nativo),
sem depender de nenhum microserviço intermediário ou instância de navegador headless.
"""

import re
import html
import json
import time
import urllib.parse
import asyncio
import logging
from typing import Dict, Any, Optional
from pathlib import Path

from app.core.config import settings

logger = logging.getLogger("cerberus.fusionsolar")

DEFAULT_KIOSK_TOKEN = "EM6wrnyAhBtOFvD7rdNu045xGpwb1bGD"
DEFAULT_HUAWEI_HOST = "intl.fusionsolar.huawei.com"


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


def _extract_kiosk_token(kiosk_url_or_token: Optional[str]) -> str:
    """
    Extrai o token 'kk' a partir de qualquer formato de URL pública ou token direto.
    Trata inclusive casos onde o caractere '#' cortou a URL no parser de .env.
    """
    if not kiosk_url_or_token:
        return DEFAULT_KIOSK_TOKEN

    raw = str(kiosk_url_or_token).strip().strip('"').strip("'")

    # Caso 1: Se for o token puro (32 caracteres alfanuméricos)
    if re.match(r"^[a-zA-Z0-9_\-]{20,50}$", raw):
        return raw

    # Caso 2: Se contiver 'kk=TOKEN'
    match = re.search(r"kk=([a-zA-Z0-9_\-]+)", raw)
    if match:
        return match.group(1)

    # Caso 3: Fallback de segurança para o token padrão conhecido
    logger.info(f"[FusionSolar] Token não localizado na string '{raw[:30]}...'. Usando token padrão.")
    return DEFAULT_KIOSK_TOKEN


def _build_huawei_rest_url(kiosk_url: str) -> str:
    """
    Constrói a URL do endpoint REST JSON nativo da Huawei a partir da URL do quiosque ou do token direto.
    Ex:
    Entrada: https://intl.fusionsolar.huawei.com/pvmswebsite/nologin/assets/build/cloud.html#/kiosk?kk=TOKEN
    Saída:   https://intl.fusionsolar.huawei.com/rest/pvms/web/kiosk/v1/station-kiosk-file?kk=TOKEN
    """
    token = None
    if getattr(settings, "fusionsolar_token", None):
        token = str(settings.fusionsolar_token).strip().strip('"').strip("'")
    if not token or len(token) < 15:
        token = _extract_kiosk_token(kiosk_url)

    host = DEFAULT_HUAWEI_HOST
    if kiosk_url:
        try:
            parsed = urllib.parse.urlparse(kiosk_url)
            if parsed.netloc:
                host = parsed.netloc
        except Exception:
            pass

    return f"https://{host}/rest/pvms/web/kiosk/v1/station-kiosk-file?kk={token}"


class FusionSolarClient:
    """
    Cliente assíncrono para extração direta de telemetria da Huawei FusionSolar.
    Executa requisição HTTP direta ao endpoint JSON oficial da Huawei com cache em memória.
    """

    def __init__(self):
        self._cache: Dict[str, Any] = {}
        self._last_fetch_ts: float = 0.0
        self._last_error: str = "Nenhum erro registrado"
        self._raw_preview: str = ""
        self._lock = asyncio.Lock()

    def _sync_http_fetch(self, url: str) -> Optional[Dict[str, Any]]:
        """
        Executa requisição HTTP usando 'requests' prioritariamente (com verify=False para
        evitar bloqueios por crypto-policies restritas do Linux Nobara/Fedora) e fallback para 'urllib'.
        """
        logger.info(f"[FusionSolar] Consultando telemetria na Huawei: {url}")

        # 1. Tentativa primária com a biblioteca 'requests'
        try:
            import requests
            import urllib3
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

            resp = requests.get(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://intl.fusionsolar.huawei.com/pvmswebsite/nologin/assets/build/cloud.html"
                },
                timeout=8.0,
                verify=False
            )
            if resp.status_code == 200:
                data = resp.json()
                self._raw_preview = str(data)[:250]
                self._last_error = ""
                return data
            else:
                self._last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
                logger.warning(f"[FusionSolar] Huawei retornou status não-200: {resp.status_code}")
        except Exception as e:
            self._last_error = f"Erro em requests: {e}"
            logger.warning(f"[FusionSolar] Falha na consulta via requests: {e}")

        # 2. Fallback secundário com urllib
        try:
            import urllib.request
            import ssl

            ctx = ssl._create_unverified_context()
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://intl.fusionsolar.huawei.com/pvmswebsite/nologin/assets/build/cloud.html"
                }
            )
            with urllib.request.urlopen(req, timeout=8.0, context=ctx) as response:
                if response.status == 200:
                    body = response.read().decode("utf-8")
                    self._last_error = ""
                    return json.loads(body)
        except Exception as e:
            self._last_error = f"Erro em urllib: {e}"
            logger.warning(f"[FusionSolar] Falha secundária na consulta via urllib: {e}")

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
                logger.warning(f"[FusionSolar] Resposta da Huawei não possui campo 'data': {raw_resp}")
                return None

            # O payload interno 'data' vem como string JSON codificada com HTML entities (&quot;)
            if isinstance(raw_data, str):
                unescaped = html.unescape(raw_data)
                station_data = json.loads(unescaped)
            elif isinstance(raw_data, dict):
                station_data = raw_data
            else:
                self._last_error = f"Tipo inesperado em 'data': {type(raw_data)}"
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

            logger.info(f"[FusionSolar] Telemetria obtida com sucesso: {pot_display} instantâneo | {daily_display} hoje ({addr})")

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
            self._last_error = f"Erro no parse de data: {e}"
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
                logger.info(f"[FusionSolar] Usando snapshot local de dados_solar.json: {parsed['potencia_realtime']}")
                return parsed

            # 4. Fallback: Cache anterior expirado
            if self._cache:
                stale = dict(self._cache)
                stale["is_stale"] = True
                return stale

            logger.warning(f"[FusionSolar] Telemetria da Huawei indisponível no momento. Detalhe: {self._last_error}")
            return {
                "available": False,
                "reason": "huawei_unreachable",
                "last_error": self._last_error,
                "potencia_realtime": "0.00 kW",
                "rendimento_hoje": "0.0 kWh"
            }

    async def get_debug_info(self) -> Dict[str, Any]:
        """Retorna informações diagnósticas detalhadas forçando a busca na Huawei."""
        fetch_result = await self.get_kiosk_data(force=True)
        rest_url = _build_huawei_rest_url(settings.fusionsolar_kiosk_url)
        return {
            "fusionsolar_enabled": settings.fusionsolar_enabled,
            "fusionsolar_token": getattr(settings, "fusionsolar_token", None),
            "fusionsolar_kiosk_url": settings.fusionsolar_kiosk_url,
            "target_rest_url": rest_url,
            "last_error": self._last_error,
            "raw_preview": self._raw_preview,
            "cache_age_seconds": round(time.time() - self._last_fetch_ts, 1) if self._last_fetch_ts else None,
            "has_cache": bool(self._cache),
            "fetch_result": fetch_result,
            "current_data": self._cache
        }


fusionsolar_client = FusionSolarClient()
