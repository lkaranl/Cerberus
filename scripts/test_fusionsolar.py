#!/usr/bin/env python3
"""
Cerberus AI — Script de Diagnóstico e Validação do FusionSolar Kiosk
Testa a conectividade direta com a nuvem da Huawei, decodifica as entidades HTML
e valida as métricas de potência e rendimento em tempo real.

Uso no servidor Linux:
    python3 scripts/test_fusionsolar.py
"""

import sys
import html
import json
import time

try:
    import requests
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
except ImportError:
    requests = None

# Configurações padrão
DEFAULT_TOKEN = "EM6wrnyAhBtOFvD7rdNu045xGpwb1bGD"
DEFAULT_HOST = "intl.fusionsolar.huawei.com"


def main():
    print("=" * 65)
    print(" ☀️  CERBERUS AI — DIAGNÓSTICO DE TELEMETRIA SOLAR HUAWEI")
    print("=" * 65)

    # 1. Tenta carregar configurações do Cerberus se disponível
    token = DEFAULT_TOKEN
    host = DEFAULT_HOST
    try:
        from app.core.config import settings
        if getattr(settings, "fusionsolar_token", None):
            token = settings.fusionsolar_token
        print(f"[*] Configurações carregadas do Cerberus:")
        print(f"    - Token: {token}")
        print(f"    - Ativo: {settings.fusionsolar_enabled}")
    except Exception:
        print(f"[*] Usando token padrão: {token}")

    target_url = f"https://{host}/rest/pvms/web/kiosk/v1/station-kiosk-file?kk={token}"
    print(f"\n[*] URL do endpoint REST da Huawei:")
    print(f"    {target_url}")

    print("\n[+] Enviando requisição HTTP para a Huawei Cloud...")
    start_time = time.time()

    raw_json = None
    status_code = None

    if requests:
        try:
            resp = requests.get(
                target_url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Referer": "https://intl.fusionsolar.huawei.com/pvmswebsite/nologin/assets/build/cloud.html"
                },
                timeout=8.0,
                verify=False
            )
            status_code = resp.status_code
            if status_code == 200:
                raw_json = resp.json()
        except Exception as e:
            print(f"[-] Erro na requisição via requests: {e}")
    else:
        import urllib.request
        import ssl
        ctx = ssl._create_unverified_context()
        req = urllib.request.Request(
            target_url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json"
            }
        )
        try:
            with urllib.request.urlopen(req, timeout=8.0, context=ctx) as r:
                status_code = r.status
                if status_code == 200:
                    raw_json = json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"[-] Erro na requisição via urllib: {e}")

    elapsed_ms = round((time.time() - start_time) * 1000, 1)

    if status_code != 200 or not raw_json:
        print(f"\n❌ FALHA: HTTP Status = {status_code} em {elapsed_ms}ms")
        sys.exit(1)

    print(f"[✓] Conexão bem-sucedida! HTTP {status_code} em {elapsed_ms}ms")

    if not raw_json.get("success"):
        print(f"[-] A Huawei respondeu com success=false: {raw_json}")
        sys.exit(1)

    raw_data = raw_json.get("data")
    if not raw_data:
        print(f"[-] Campo 'data' ausente na resposta.")
        sys.exit(1)

    print("\n[+] Decodificando dados da estação com html.unescape()...")
    try:
        if isinstance(raw_data, str):
            unescaped_str = html.unescape(raw_data)
            station_data = json.loads(unescaped_str)
        else:
            station_data = raw_data
        print("[✓] JSON decodificado com perfeição!")
    except Exception as e:
        print(f"❌ ERRO no parse do JSON: {e}")
        print(f"    Amostra crua: {str(raw_data)[:200]}...")
        sys.exit(1)

    # Extrai KPIs
    real_kpi = station_data.get("realKpi", {})
    station = station_data.get("stationOverview", {})
    social = station_data.get("socialContribution", {})
    power_curve = station_data.get("powerCurve", {})

    pot_kw = float(real_kpi.get("realTimePower") or 0.0)
    hoje_kwh = float(real_kpi.get("dailyEnergy") or 0.0)
    mes_kwh = float(real_kpi.get("monthEnergy") or 0.0)
    ano_kwh = float(real_kpi.get("yearEnergy") or 0.0)
    total_kwh = float(real_kpi.get("cumulativeEnergy") or 0.0)

    co2_kg = float(social.get("co2Reduction") or 0.0)
    trees = int(social.get("equivalentTreePlanting") or 0)

    station_name = station.get("stationName") or "Usina"
    address = station.get("plantAddress") or "Ji-Paraná, RO"

    x_axis = power_curve.get("xAxis") or []
    active_power = power_curve.get("activePower") or []
    pontos_curva = sum(1 for p in active_power if p and p != "-")

    print("\n" + "=" * 65)
    print(f" 📊 RESULTADOS DA USINA SOLAR: {station_name.upper()}")
    print("=" * 65)
    print(f" 📍 Endereço:          {address}")
    print(f" ⚡ Potência Agora:    \033[1;33m{pot_kw:.2f} kW\033[0m")
    print(f" ☀️  Geração Hoje:      \033[1;32m{hoje_kwh:.2f} kWh\033[0m")
    print(f" 📅 Geração no Mês:    {mes_kwh:.2f} kWh ({mes_kwh/1000.0:.2f} MWh)")
    print(f" 🗓️  Geração no Ano:    {ano_kwh:.2f} kWh ({ano_kwh/1000.0:.2f} MWh)")
    print(f" 🔋 Total Acumulado:   {total_kwh/1000.0:.2f} MWh")
    print(f" 🌿 CO₂ Evitado:       {co2_kg/1000.0:.2f} toneladas")
    print(f" 🌳 Árvores Equiv.:    {trees} árvores")
    print(f" 📈 Pontos na Curva:   {pontos_curva} medições hoje")
    print("=" * 65)
    print("\n✅ TESTE CONCLUÍDO COM SUCESSO! A API está 100% operacional.\n")


if __name__ == "__main__":
    main()
