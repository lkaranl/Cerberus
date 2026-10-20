import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx

from app.core.config import settings
from app.core.logging import logger
from app.ai.risk_evaluator import risk_evaluator

class TelegramBot:
    """
    Instância desacoplada de um bot do Telegram.
    Pode operar como Bot 1 (Feed Geral) ou Bot 2 (Alertas Críticos / Prioritários).
    """
    def __init__(
        self,
        name: str = "CerberusBot",
        token_getter=None,
        chat_id_getter=None,
        enabled_getter=None,
        is_critical_channel: bool = False
    ):
        self.name = name
        self.token_getter = token_getter
        self.chat_id_getter = chat_id_getter
        self.enabled_getter = enabled_getter
        self.is_critical_channel = is_critical_channel

        self.client: Optional[httpx.AsyncClient] = None
        self.poll_task: Optional[asyncio.Task] = None
        self.is_active: bool = False
        self.bot_username: str = ""
        self.base_url: str = ""
        self._last_update_id: int = 0
        self._running: bool = False
        self._sent_event_ids: set = set()

    async def start(self) -> bool:
        """Inicializa o cliente HTTP e inicia o serviço se estiver habilitado no .env."""
        is_enabled = self.enabled_getter() if self.enabled_getter else False
        if not is_enabled:
            logger.info(f"ℹ️ Módulo Telegram [{self.name}] desativado no .env.")
            return False

        token = (self.token_getter() if self.token_getter else "").strip()
        if not token:
            logger.warning(f"⚠️ Módulo Telegram [{self.name}] ativado, mas o TOKEN está vazio no .env.")
            return False

        self.base_url = f"https://api.telegram.org/bot{token}"
        self.client = httpx.AsyncClient(timeout=40.0)

        try:
            resp = await self.client.get(f"{self.base_url}/getMe")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    self.bot_username = data["result"].get("username", self.name)
                    self.is_active = True
                    self._running = True
                    logger.info(f"🤖 Bot do Telegram [{self.name}] autenticado com sucesso: @{self.bot_username}")

                    await self._register_commands()
                    self.poll_task = asyncio.create_task(self._poll_updates())
                    return True
            logger.error(f"Falha ao autenticar bot [{self.name}]: {resp.text}")
        except Exception as err:
            logger.error(f"Erro de conexão com API do Telegram [{self.name}]: {err}")

        self.is_active = False
        return False

    async def stop(self) -> None:
        """Finaliza suavemente conexões e tarefas de polling."""
        self._running = False
        if self.poll_task and not self.poll_task.done():
            self.poll_task.cancel()
            try:
                await self.poll_task
            except asyncio.CancelledError:
                pass

        if self.client:
            await self.client.aclose()
            self.client = None

        self.is_active = False
        logger.info(f"Módulo Telegram [{self.name}] encerrado.")

    async def _register_commands(self) -> None:
        if not self.client:
            return
        commands = [
            {"command": "status", "description": "Ver status do sistema e telemetria"},
            {"command": "snapshot", "description": "Foto instantânea das câmeras ao vivo"},
            {"command": "alertas", "description": "Ver os últimos eventos detectados"},
            {"command": "ajuda", "description": "Menu de comandos e configurações"}
        ]
        try:
            await self.client.post(f"{self.base_url}/setMyCommands", json={"commands": commands})
        except Exception as err:
            logger.debug(f"Não foi possível registrar comandos no Telegram [{self.name}]: {err}")

    async def send_alert(self, event: Dict[str, Any]) -> bool:
        """
        Envia notificação de alerta para o Telegram de forma não-bloqueante.
        Se este for o Bot Crítico, valida com o Avaliador de IA se o risco é 'alto'.
        """
        if not self.is_active or not self.client:
            return False

        chat_id = (self.chat_id_getter() if self.chat_id_getter else "").strip()
        if not chat_id:
            logger.debug(f"Alerta não enviado ao Telegram [{self.name}]: CHAT_ID não definido.")
            return False

        entity_type = event.get("entity_type", "person")
        if entity_type == "person" and not settings.telegram_notify_person:
            return False
        if entity_type == "vehicle" and not settings.telegram_notify_vehicle:
            return False
        if entity_type == "pet" and not settings.telegram_notify_pet:
            return False

        event_id = event.get("event_id", "")
        if event_id and event_id in self._sent_event_ids:
            return False

        is_upgrade = (event.get("_msg_type") == "EVENT_UPDATED")
        has_face = event.get("has_face_captured", False)

        # Regra de foco no rosto para pessoas
        if entity_type == "person" and not has_face and not is_upgrade:
            return False

        # Avaliação de IA Contextual (OpenRouter / Heurística Local)
        # Se for o Bot Crítico, só envia se o risco for "alto"
        risk_data = await risk_evaluator.evaluate_risk(event)
        event["risk_analysis"] = risk_data
        nivel_risco = risk_data.get("risco", "baixo").lower()

        if self.is_critical_channel:
            if nivel_risco != "alto":
                logger.debug(
                    f"🛡️ [Bot Crítico] Evento {event_id} filtrado (risco={nivel_risco}). "
                    f"Enviado apenas ao feed geral."
                )
                return False

        if is_upgrade:
            return await self._send_upgrade_notice(event, chat_id, risk_data)

        # Formatação do card do Telegram
        category_icons = {"person": "👤", "vehicle": "🚗", "pet": "🐾"}
        icon = category_icons.get(entity_type, "🎯")
        name = event.get("person_name", "Desconhecido")
        is_known = event.get("is_known", False)
        known_badge = "✅ Reconhecido" if is_known else "⚠️ Desconhecido"
        face_conf = event.get("face_confidence", 0.0)
        face_pct = f" ({int(face_conf * 100)}%)" if face_conf > 0 else ""

        plate = event.get("license_plate", "")
        plate_str = f"\n🏷️ <b>Placa:</b> <code>{plate}</code>" if plate else ""

        ts_str = event.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(ts_str)
            hora_fmt = dt.strftime("%d/%m/%Y às %H:%M:%S")
        except Exception:
            hora_fmt = ts_str

        # Destaque de IA
        motivo = risk_data.get("motivo", "")
        acao = risk_data.get("acao", "")
        avaliador = risk_data.get("avaliador", "IA")

        if self.is_critical_channel or nivel_risco == "alto":
            titulo = "🚨🚨 <b>ALERTA CRÍTICO DE SEGURANÇA - CERBERUS</b> 🚨🚨"
            risco_badge = "🔴 <b>RISCO ALTO (CRÍTICO)</b>"
        elif nivel_risco == "medio":
            titulo = "⚠️ <b>ALERTA DE ATENÇÃO - CERBERUS</b>"
            risco_badge = "🟡 <b>RISCO MÉDIO</b>"
        else:
            titulo = "🔔 <b>NOTIFICAÇÃO DE ROTINA - CERBERUS</b>"
            risco_badge = "🟢 <b>RISCO BAIXO</b>"

        caption = (
            f"{titulo}\n\n"
            f"📹 <b>Câmera:</b> {event.get('camera_name', 'Câmera')}\n"
            f"{icon} <b>Alvo:</b> {name}{face_pct} ({known_badge}){plate_str}\n"
            f"🕒 <b>Horário:</b> {hora_fmt}\n"
            f"🛡️ <b>Avaliação:</b> {risco_badge}\n"
            f"🧠 <b>Motivo IA:</b> <i>{motivo}</i>\n"
            f"👉 <b>Recomendação:</b> <b>{acao}</b>\n\n"
            f"🎯 <b>Confiança da IA:</b> {int(event.get('confidence', 0.0) * 100)}% | 🆔 <code>{event_id}</code>"
        )

        crop_path = event.get("crop_path")
        sent = False
        if crop_path and Path(crop_path).exists():
            # No Bot Crítico, som alto e prioritário (disable_notification=False)
            disable_notification = not self.is_critical_channel
            sent = await self._send_single_photo(
                chat_id,
                Path(crop_path),
                caption,
                disable_notification=disable_notification
            )

        if sent and event_id:
            self._sent_event_ids.add(event_id)
            if len(self._sent_event_ids) > 250:
                self._sent_event_ids = set(list(self._sent_event_ids)[-125:])

        return sent

    async def _send_upgrade_notice(self, event: Dict[str, Any], chat_id: str, risk_data: Dict[str, Any]) -> bool:
        """Envia a foto do rosto quando a pessoa que estava de costas vira para a câmera."""
        event_id = event.get("event_id", "")
        if event_id and event_id in self._sent_event_ids:
            return False

        name = event.get("person_name", "Desconhecido")
        face_conf = event.get("face_confidence", 0.0)
        face_pct = f" ({int(face_conf * 100)}%)" if face_conf > 0 else ""
        is_known = event.get("is_known", False)
        known_badge = "✅ Reconhecido" if is_known else "⚠️ Desconhecido"

        motivo = risk_data.get("motivo", "")
        acao = risk_data.get("acao", "")

        caption = (
            f"👤 <b>ROSTO IDENTIFICADO - CERBERUS</b>\n\n"
            f"📹 <b>Câmera:</b> {event.get('camera_name', 'Câmera')}\n"
            f"🎯 <b>Alvo:</b> <b>{name}</b>{face_pct} ({known_badge})\n"
            f"🕒 <b>Horário:</b> {datetime.now().strftime('%d/%m/%Y às %H:%M:%S')}\n"
            f"🧠 <b>Análise de Risco:</b> {risk_data.get('risco', 'baixo').upper()} - <i>{motivo}</i>\n"
            f"🆔 <b>Evento:</b> <code>{event_id}</code>\n"
            f"✨ <i>Rosto capturado com sucesso ao virar para a câmera.</i>"
        )

        crop_path = event.get("crop_path")
        sent = False
        if crop_path and Path(crop_path).exists():
            sent = await self._send_single_photo(chat_id, Path(crop_path), caption)

        if sent and event_id:
            self._sent_event_ids.add(event_id)
            if len(self._sent_event_ids) > 250:
                self._sent_event_ids = set(list(self._sent_event_ids)[-125:])

        return sent

    async def _send_single_photo(
        self,
        chat_id: str,
        photo_path: Path,
        caption: str,
        disable_notification: bool = False
    ) -> bool:
        """Envia uma única foto com legenda formatada em HTML."""
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": (photo_path.name, f, "image/jpeg")}
                data = {
                    "chat_id": chat_id,
                    "caption": caption,
                    "parse_mode": "HTML",
                    "disable_notification": disable_notification
                }
                resp = await self.client.post(f"{self.base_url}/sendPhoto", data=data, files=files)
                return resp.status_code == 200
        except Exception as err:
            logger.debug(f"Erro ao enviar foto para o Telegram [{self.name}]: {err}")
            return False

    async def _send_text_message(self, chat_id: str, text: str) -> bool:
        """Envia uma mensagem de texto simples formatada em HTML."""
        try:
            data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            resp = await self.client.post(f"{self.base_url}/sendMessage", data=data)
            return resp.status_code == 200
        except Exception as err:
            logger.debug(f"Erro ao enviar mensagem de texto [{self.name}]: {err}")
            return False

    async def _poll_updates(self) -> None:
        """Long-polling assíncrono para processar comandos recebidos do usuário."""
        logger.info(f"📡 Iniciando escuta de comandos via Telegram [{self.name}]...")
        while self._running:
            try:
                params = {
                    "offset": self._last_update_id + 1,
                    "timeout": 20,
                    "allowed_updates": ["message"]
                }
                resp = await self.client.get(f"{self.base_url}/getUpdates", params=params, timeout=30.0)
                if resp.status_code == 200:
                    data = resp.json()
                    for update in data.get("result", []):
                        self._last_update_id = update["update_id"]
                        if "message" in update:
                            await self._handle_message(update["message"])
            except asyncio.CancelledError:
                break
            except Exception as err:
                await asyncio.sleep(5.0)

    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Roteia comandos de texto recebidos para os manipuladores específicos."""
        chat = message.get("chat", {})
        chat_id = str(chat.get("id", ""))
        text = message.get("text", "").strip()
        user = message.get("from", {})
        user_name = user.get("first_name", "Usuário")

        configured_chat = (self.chat_id_getter() if self.chat_id_getter else "").strip()
        if not configured_chat:
            env_var = "TELEGRAM_CRITICAL_CHAT_ID" if self.is_critical_channel else "TELEGRAM_CHAT_ID"
            welcome = (
                f"👋 Olá, {user_name}!\n\n"
                f"Seu <b>Chat ID</b> é: <code>{chat_id}</code>\n\n"
                f"👉 Para autorizar este chat no bot [{self.name}], adicione no seu <code>.env</code>:\n"
                f"<code>{env_var}={chat_id}</code>\n"
                f"E reinicie o serviço com <code>sudo systemctl restart cerberus</code>."
            )
            await self._send_text_message(chat_id, welcome)
            return

        if chat_id != configured_chat:
            logger.warning(f"🚫 Comando não autorizado no Telegram [{self.name}] do chat {chat_id} ({user_name})")
            await self._send_text_message(chat_id, "⛔ <b>Acesso Negado.</b> Este bot é privado do sistema Cerberus.")
            return

        cmd = text.split()[0].lower() if text else ""
        if "@" in cmd:
            cmd = cmd.split("@")[0]

        if cmd in ("/start", "/ajuda", "/help"):
            await self._cmd_help(chat_id, user_name)
        elif cmd in ("/status", "/telemetria"):
            await self._cmd_status(chat_id)
        elif cmd in ("/snapshot", "/foto", "/snap"):
            await self._cmd_snapshot(chat_id)
        elif cmd in ("/alertas", "/eventos"):
            await self._cmd_alerts(chat_id)
        else:
            await self._send_text_message(
                chat_id,
                "Comando não reconhecido. Digite <b>/ajuda</b> para ver as opções disponíveis."
            )

    async def _cmd_help(self, chat_id: str, user_name: str) -> None:
        canal = "CANAL DE ALERTAS CRÍTICOS (ALTO RISCO)" if self.is_critical_channel else "FEED GERAL DE AUDITORIA"
        help_text = (
            f"🛡️ <b>CERBERUS - {canal}</b>\n"
            f"<i>Sistema Autônomo de Vigilância & Visão Computacional</i>\n\n"
            f"Olá, <b>{user_name}</b>! Estou monitorando sua residência 24/7.\n\n"
            f"📋 <b>Comandos Disponíveis:</b>\n"
            f"🔹 <b>/status</b> - Telemetria do servidor, GPU e FPS das câmeras\n"
            f"🔹 <b>/snapshot</b> - Captura fotos das câmeras ao vivo agora\n"
            f"🔹 <b>/alertas</b> - Lista os últimos 5 eventos registrados\n"
            f"🔹 <b>/ajuda</b> - Exibe esta mensagem de suporte\n"
        )
        await self._send_text_message(chat_id, help_text)

    async def _cmd_status(self, chat_id: str) -> None:
        from app.stream.manager import stream_manager
        cams_info = []
        for cam in stream_manager.cameras_config:
            if cam.active:
                fps = stream_manager.inference_fps.get(cam.id, 0.0)
                status_icon = "🟢" if fps > 0 else "🟡"
                cams_info.append(f"  {status_icon} <b>{cam.name}</b>: {fps:.1f} FPS")

        cams_str = "\n".join(cams_info) if cams_info else "  Nenhuma câmera ativa."

        try:
            import torch
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
            vram_mb = int(torch.cuda.memory_allocated(0) / (1024 * 1024)) if torch.cuda.is_available() else 0
        except Exception:
            gpu_name = "NVIDIA GPU"
            vram_mb = 0

        engine_yolo = "TensorRT FP16 ⚡" if settings.use_tensorrt else "PyTorch"
        engine_face = "TensorRT FP16 🚀" if settings.use_tensorrt else "PyTorch"

        status_msg = (
            f"📊 <b>TELEMETRIA DO SISTEMA CERBERUS</b>\n\n"
            f"📹 <b>Câmeras Ativas:</b>\n{cams_str}\n\n"
            f"⚙️ <b>Aceleração de IA:</b>\n"
            f"  • Detector YOLO: <code>{engine_yolo}</code>\n"
            f"  • FaceNet Facial: <code>{engine_face}</code>\n"
            f"  • Dispositivo: <code>{gpu_name}</code> (VRAM: {vram_mb} MB)\n\n"
            f"🤖 <b>Avaliador Contextual:</b> <code>{settings.openrouter_model}</code>\n"
            f"🕒 <b>Hora do Servidor:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"✅ <b>Status:</b> Operação contínua 24/7 sem interrupções."
        )
        await self._send_text_message(chat_id, status_msg)

    async def _cmd_snapshot(self, chat_id: str) -> None:
        from app.stream.manager import stream_manager
        await self._send_text_message(chat_id, "📸 Capturando fotos ao vivo das câmeras, aguarde...")

        sent_count = 0
        for cam_id, jpeg_bytes in stream_manager.latest_annotated_jpeg.items():
            cam_name = cam_id
            for c in stream_manager.cameras_config:
                if c.id == cam_id:
                    cam_name = c.name
                    break

            try:
                caption = f"📹 <b>Ao Vivo:</b> {cam_name}\n🕒 {datetime.now().strftime('%H:%M:%S')}"
                files = {"photo": (f"{cam_id}.jpg", jpeg_bytes, "image/jpeg")}
                data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
                resp = await self.client.post(f"{self.base_url}/sendPhoto", data=data, files=files)
                if resp.status_code == 200:
                    sent_count += 1
            except Exception as err:
                logger.debug(f"Erro ao enviar snapshot de {cam_name}: {err}")

        if sent_count == 0:
            await self._send_text_message(chat_id, "⚠️ Não há frames recentes disponíveis nas câmeras no momento.")

    async def _cmd_alerts(self, chat_id: str) -> None:
        from app.storage.database import db
        recent = await db.get_recent_events(limit=5)
        if not recent:
            await self._send_text_message(chat_id, "ℹ️ Nenhum alerta registrado no histórico recente.")
            return

        lines = ["📜 <b>ÚLTIMOS 5 ALERTAS REGISTRADOS:</b>\n"]
        for evt in recent:
            ts = evt.get("timestamp", "")
            try:
                hora = datetime.fromisoformat(ts).strftime("%d/%m %H:%M")
            except Exception:
                hora = ts[:16]

            entity = evt.get("entity_type", "person")
            icon = "👤" if entity == "person" else ("🚗" if entity == "vehicle" else "🐾")
            name = evt.get("person_name", "Desconhecido")
            cam = evt.get("camera_name", "Câmera")
            lines.append(f"{icon} <b>{name}</b> na <i>{cam}</i> às {hora}")

        lines.append("\n👉 Acesse o painel web para ver os recortes e gravações completas.")
        await self._send_text_message(chat_id, "\n".join(lines))


# Instância do Bot 1 (Feed Geral / Silencioso)
telegram_bot = TelegramBot(
    name="FeedGeralBot",
    token_getter=lambda: settings.telegram_bot_token,
    chat_id_getter=lambda: settings.telegram_chat_id,
    enabled_getter=lambda: settings.telegram_enabled,
    is_critical_channel=False
)

# Instância do Bot 2 (Alertas Críticos / Prioritários com Som)
telegram_critical_bot = TelegramBot(
    name="CriticoBot",
    token_getter=lambda: settings.telegram_critical_bot_token,
    chat_id_getter=lambda: settings.telegram_critical_chat_id,
    enabled_getter=lambda: settings.telegram_critical_enabled,
    is_critical_channel=True
)
