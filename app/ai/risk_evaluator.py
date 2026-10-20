import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

import httpx

from app.core.config import settings
from app.core.logging import logger

SYSTEM_PROMPT = """Você é um auditor sênior de segurança patrimonial e residencial autônomo (Cerberus).
Sua missão é avaliar o risco de um evento de segurança detectado por câmeras e classificá-lo rigorosamente.

Regras de Contexto:
1. Pessoas CONHECIDAS (moradores) normalmente são risco "baixo", salvo comportamento bizarro ou horário extremo em áreas restritas.
2. Pessoas DESCONHECIDAS:
   - Em horário diurno (07:00 às 21:00) na calçada/portão de entrada geralmente é risco "baixo" ou "medio".
   - Em horário noturno / madrugada (22:00 às 06:00), especialmente em áreas como "fundo", "garagem", "quintal", "lateral" ou com permanência elevada (> 15s), é risco "alto" (CRÍTICO).
   - Tentativa de aproximação persistente em portão na madrugada é risco "alto".
3. Veículos:
   - Carros passando na rua ou parados rapidamente de dia = risco "baixo".
   - Veículos estranhos parados de madrugada diante do portão por muito tempo = risco "medio" ou "alto".
4. Animais: quase sempre risco "baixo", a não ser predador/situação anormal.

Responda ESTRITAMENTE em formato JSON puro, sem crases de markdown e sem texto adicional:
{
  "risco": "baixo" | "medio" | "alto",
  "motivo": "frase curta e objetiva em português explicando o motivo",
  "acao": "recomendação curta de ação para o morador"
}
"""

class RiskEvaluator:
    """
    Avaliador Inteligente de Risco Contextual integrado à OpenRouter.
    Opera sob demanda por evento (Event-Driven) com consumo ultrabaixo de tokens (~150 tokens por chamada).
    Possui fallback heurístico local e instantâneo caso a chave da API não esteja preenchida ou ocorra timeout.
    """
    def __init__(self):
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"

    def _build_event_payload(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Extrai apenas os metadados contextuais essenciais (~80-100 tokens)."""
        camera_name = event.get("camera_name", "Câmera")
        entity_type = event.get("entity_type", "person")
        person_name = event.get("person_name", "Desconhecido")
        is_known = bool(event.get("is_known", False))
        
        # Horário local
        ts_str = event.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(ts_str)
            hora_str = dt.strftime("%H:%M")
            hora_int = dt.hour
        except Exception:
            hora_str = ts_str
            hora_int = 12

        confidence = event.get("confidence", 0.0)
        face_confidence = event.get("face_confidence", 0.0)
        plate = event.get("license_plate", "")
        duration_sec = event.get("duration_seconds", 5)

        return {
            "camera": camera_name,
            "horario": hora_str,
            "periodo_noturno": (hora_int >= 22 or hora_int < 6),
            "tipo": entity_type,
            "identidade": person_name,
            "morador_conhecido": is_known,
            "placa_veiculo": plate if plate else None,
            "tempo_na_cena_seg": duration_sec,
            "confianca_reconhecimento": round(face_confidence, 2) if face_confidence else None
        }

    def _fallback_heuristic(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Heurística local de segurança (zero latência e 100% offline).
        Garante classificação imediata caso a OpenRouter esteja indisponível ou sem API Key.
        """
        is_known = payload.get("morador_conhecido", False)
        is_night = payload.get("periodo_noturno", False)
        cam = payload.get("camera", "").lower()
        entity = payload.get("tipo", "person")
        name = payload.get("identidade", "Desconhecido")

        if is_known:
            return {
                "risco": "baixo",
                "motivo": f"Morador autorizado ({name}) detectado.",
                "acao": "Nenhuma ação necessária."
            }

        # Desconhecido de madrugada nos fundos, garagem ou portão
        if entity == "person" and is_night:
            if "fundo" in cam or "garagem" in cam or "quintal" in cam or "lateral" in cam:
                return {
                    "risco": "alto",
                    "motivo": "Presença de pessoa desconhecida em área perimetral restrita durante a madrugada.",
                    "acao": "Verifique as câmeras imediatamente ou acione iluminação."
                }
            return {
                "risco": "alto",
                "motivo": "Pessoa desconhecida detectada na madrugada.",
                "acao": "Acompanhe ao vivo pelo monitoramento."
            }

        if entity == "person" and not is_night:
            if "fundo" in cam:
                return {
                    "risco": "medio",
                    "motivo": "Pessoa não identificada na área do fundo em horário diurno.",
                    "acao": "Verificar se é prestador de serviço ou visitante."
                }
            return {
                "risco": "baixo",
                "motivo": "Movimento habitual de pedestre/visitante durante o dia.",
                "acao": "Monitoramento rotineiro."
            }

        return {
            "risco": "baixo",
            "motivo": "Detecção de rotina sem indicadores de ameaça perimetral.",
            "acao": "Manter monitoramento padrão."
        }

    async def evaluate_risk(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa o evento via OpenRouter LLM ou aciona o fallback heurístico.
        Retorna dicionário com:
        - risco: 'baixo' | 'medio' | 'alto'
        - motivo: explicação resumida
        - acao: recomendação prática
        - avaliador: 'openrouter' ou 'heuristica_local'
        """
        payload = self._build_event_payload(event)
        api_key = settings.openrouter_api_key.strip()

        # Se não há chave de API cadastrada, usa a heurística local rápida
        if not api_key:
            res = self._fallback_heuristic(payload)
            res["avaliador"] = "heuristica_local"
            return res

        model = settings.openrouter_model or "google/gemini-2.5-flash"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/karan/cerberus",
            "X-Title": "Cerberus Security AI"
        }

        user_content = f"Avalie este evento de segurança residencial:\n{json.dumps(payload, ensure_ascii=False)}"

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,
            "max_tokens": 400
        }

        try:
            timeout = float(settings.openrouter_timeout or 15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.api_url, headers=headers, json=data)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    msg = resp_json.get("choices", [{}])[0].get("message", {})
                    raw_text = (msg.get("content") or "").strip()
                    # Limpeza de blocos markdown caso o modelo insira
                    if raw_text.startswith("```"):
                        raw_text = raw_text.strip("`").replace("json", "").strip()
                    if raw_text:
                        parsed = json.loads(raw_text)
                        parsed["avaliador"] = f"openrouter:{model}"
                        return parsed
                else:
                    logger.warning(f"⚠️ OpenRouter retornou HTTP {resp.status_code}: {resp.text[:120]}. Usando fallback.")
        except Exception as err:
            logger.debug(f"Erro na chamada OpenRouter: {err}. Acionando fallback heurístico.")

        # Fallback de segurança garantido
        fallback = self._fallback_heuristic(payload)
        fallback["avaliador"] = "heuristica_local_fallback"
        return fallback

    async def answer_security_question(self, question: str, recent_events: list) -> str:
        """
        Responde a perguntas em linguagem natural feitas pelo morador no Telegram,
        analisando os eventos recentes do banco SQLite.
        """
        api_key = settings.openrouter_api_key.strip()
        if not api_key:
            return (
                "⚠️ A chave da OpenRouter (OPENROUTER_API_KEY) não está configurada no .env. "
                "Configure para habilitar o assistente de IA."
            )

        model = settings.openrouter_model or "google/gemini-2.5-flash"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/karan/cerberus",
            "X-Title": "Cerberus Security AI"
        }

        # Formata o resumo compacto dos eventos (~150-200 tokens)
        events_summary = []
        for e in recent_events[:30]:
            ts = e.get("timestamp", "")
            try:
                hora = datetime.fromisoformat(ts).strftime("%H:%M")
            except Exception:
                hora = ts[:16]
            cam = e.get("camera_name", "Câmera")
            name = e.get("person_name", "Desconhecido")
            etype = e.get("entity_type", "person")
            plate = e.get("license_plate", "")
            plate_info = f" [Placa: {plate}]" if plate else ""
            events_summary.append(f"- {hora} | {cam} | {etype}: {name}{plate_info}")

        events_text = "\n".join(events_summary) if events_summary else "Nenhum evento registrado recentemente."

        prompt_system = (
            "Você é o assistente inteligente de segurança residencial Cerberus (Sentinela Vision).\n"
            "O morador fez uma pergunta sobre o que aconteceu nas câmeras.\n"
            "Responda de forma direta, prestativa e humana em português do Brasil, "
            "baseando-se ESTRITAMENTE na lista de eventos fornecida.\n"
            "Se a informação não estiver na lista, informe educadamente que nada foi detectado no período."
        )

        user_content = (
            f"Histórico recente de eventos nas câmeras:\n{events_text}\n\n"
            f"Pergunta do morador: {question}"
        )

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }

        try:
            timeout = float(settings.openrouter_timeout or 15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.api_url, headers=headers, json=data)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    msg = resp_json.get("choices", [{}])[0].get("message", {})
                    content = (msg.get("content") or "").strip()
                    if content:
                        return content
                logger.warning(f"⚠️ OpenRouter erro no chat: {resp.status_code} - {resp.text[:120]}")
        except Exception as err:
            logger.debug(f"Erro ao consultar assistente de IA: {err}")

        return "Desculpe, ocorreu uma instabilidade temporária na consulta à IA. Tente novamente em instantes."

    async def generate_daily_briefing(self, analytics: dict, recent_events: list) -> str:
        """Gera um briefing executivo consolidado da segurança residencial."""
        api_key = settings.openrouter_api_key.strip()
        if not api_key:
            return "⚠️ OpenRouter não configurada para gerar o briefing."

        model = settings.openrouter_model or "google/gemini-2.5-flash"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/karan/cerberus",
            "X-Title": "Cerberus Security AI"
        }

        prompt_system = (
            "Você é o auditor de segurança residencial Cerberus.\n"
            "Gere um Briefing de Segurança diário conciso em tópicos no formato HTML para o Telegram.\n"
            "Destaque: status geral do perímetro, moradores vistos, veículos e se houve qualquer movimentação anormal na madrugada."
        )

        user_content = f"Dados analíticos: {json.dumps(analytics, ensure_ascii=False)}\nEventos recentes: {json.dumps(recent_events[:25], ensure_ascii=False)}"

        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": prompt_system},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.2,
            "max_tokens": 500
        }

        try:
            timeout = float(settings.openrouter_timeout or 15.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(self.api_url, headers=headers, json=data)
                if resp.status_code == 200:
                    resp_json = resp.json()
                    msg = resp_json.get("choices", [{}])[0].get("message", {})
                    content = (msg.get("content") or "").strip()
                    if content:
                        return content
        except Exception as err:
            logger.debug(f"Erro no briefing IA: {err}")

        return "Não foi possível gerar o briefing no momento."

risk_evaluator = RiskEvaluator()

