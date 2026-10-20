import os
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

import httpx

from app.core.config import settings
from app.core.logging import logger

class TelegramBot:
    """
    Módulo desacoplado de integração com Telegram.
    Opera 100% assíncrono para notificações de alerta em tempo real e
    comandos de telemetria/snapshot remoto sem impactar a inferência de IA.
    """
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        self.poll_task: Optional[asyncio.Task] = None
        self.is_active: bool = False
        self.bot_username: str = ""
        self.base_url: str = ""
        self._last_update_id: int = 0
        self._running: bool = False

    async def start(self) -> bool:
        """
        Inicializa o cliente HTTP e inicia o serviço se estiver habilitado no .env.
        Se não estiver habilitado, não gera erros e opera em modo inativo.
        """
        if not settings.telegram_enabled:
            logger.info("ℹ️ Módulo Telegram desativado (TELEGRAM_ENABLED=false).")
            return False

        token = settings.telegram_bot_token.strip()
        if not token:
            logger.warning("⚠️ Módulo Telegram ativado, mas TELEGRAM_BOT_TOKEN está vazio no .env.")
            return False

        self.base_url = f"https://api.telegram.org/bot{token}"
        self.client = httpx.AsyncClient(timeout=40.0)

        # Valida credenciais com a API do Telegram
        try:
            resp = await self.client.get(f"{self.base_url}/getMe")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("ok"):
                    self.bot_username = data["result"].get("username", "CerberusBot")
                    self.is_active = True
                    self._running = True
                    logger.info(f"🤖 Bot do Telegram autenticado com sucesso: @{self.bot_username}")

                    # Configura lista de comandos visíveis no menu do Telegram
                    await self._register_commands()

                    # Inicia tarefa em segundo plano para escutar comandos (long polling)
                    self.poll_task = asyncio.create_task(self._poll_updates())
                    return True
            logger.error(f"Falha ao autenticar bot do Telegram: {resp.text}")
        except Exception as err:
            logger.error(f"Erro de conexão com a API do Telegram: {err}")

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
        logger.info("Módulo Telegram encerrado.")

    async def _register_commands(self) -> None:
        """Registra a lista de comandos no menu interativo do Telegram."""
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
            logger.debug(f"Não foi possível registrar comandos no Telegram: {err}")

    # =========================================================================
    # ENVIO DE ALERTAS E EVIDÊNCIAS
    # =========================================================================

    async def send_alert(self, event: Dict[str, Any]) -> bool:
        """
        Envia notificação de alerta com evidências fotográficas para o Telegram.
        Executado de forma totalmente não-bloqueante.
        """
        if not self.is_active or not self.client:
            return False

        chat_id = settings.telegram_chat_id.strip()
        if not chat_id:
            logger.debug("Alerta não enviado ao Telegram: TELEGRAM_CHAT_ID não definido.")
            return False

        entity_type = event.get("entity_type", "person")

        # Filtros por categoria configurados no .env
        if entity_type == "person" and not settings.telegram_notify_person:
            return False
        if entity_type == "vehicle" and not settings.telegram_notify_vehicle:
            return False
        if entity_type == "pet" and not settings.telegram_notify_pet:
            return False

        # Caso especial: Upgrade de Evidência (a pessoa mostrou o rosto após alerta inicial de costas)
        if event.get("_msg_type") == "EVENT_UPDATED":
            return await self._send_upgrade_notice(event, chat_id)

        # Formatação do novo alerta
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

        caption = (
            f"🚨 <b>ALERTA DE SEGURANÇA - CERBERUS</b>\n\n"
            f"📹 <b>Câmera:</b> {event.get('camera_name', 'Câmera')}\n"
            f"{icon} <b>Alvo:</b> {name}{face_pct} ({known_badge}){plate_str}\n"
            f"🕒 <b>Horário:</b> {hora_fmt}\n"
            f"🎯 <b>Confiança da IA:</b> {int(event.get('confidence', 0.0) * 100)}%\n"
            f"🆔 <b>Evento:</b> <code>{event.get('event_id', '')}</code>"
        )

        crop_path = event.get("crop_path")
        full_path = event.get("full_path")

        # Tenta enviar álbum completo (recorte detalhado + cena inteira)
        if settings.telegram_send_both_photos and crop_path and full_path:
            p_crop = Path(crop_path)
            p_full = Path(full_path)
            if p_crop.exists() and p_full.exists():
                return await self._send_media_group(chat_id, p_crop, p_full, caption)

        # Fallback para envio de foto única (prioriza o recorte do alvo)
        target_img = crop_path if (crop_path and Path(crop_path).exists()) else full_path
        if target_img and Path(target_img).exists():
            return await self._send_single_photo(chat_id, Path(target_img), caption)

        # Se não houver foto em disco, envia mensagem de texto
        return await self._send_text_message(chat_id, caption)

    async def _send_upgrade_notice(self, event: Dict[str, Any], chat_id: str) -> bool:
        """Envia aviso quando o sistema capturou o rosto nítido de uma pessoa vista antes de costas."""
        name = event.get("person_name", "Desconhecido")
        face_conf = event.get("face_confidence", 0.0)
        face_pct = f" ({int(face_conf * 100)}%)" if face_conf > 0 else ""

        caption = (
            f"📸 <b>UPGRADE FACIAL - ROSTO IDENTIFICADO!</b>\n\n"
            f"📹 <b>Câmera:</b> {event.get('camera_name', 'Câmera')}\n"
            f"👤 <b>Identidade Confirmada:</b> <b>{name}</b>{face_pct}\n"
            f"🆔 <b>Evento Atualizado:</b> <code>{event.get('event_id', '')}</code>\n"
            f"✨ <i>O recorte da nuca foi substituído pela foto frontal do rosto.</i>"
        )

        # Se houver recorte recente salvo, envia a nova foto do rosto
        crop_path = event.get("crop_path")
        if crop_path and Path(crop_path).exists():
            return await self._send_single_photo(chat_id, Path(crop_path), caption)

        return await self._send_text_message(chat_id, caption)

    async def _send_single_photo(self, chat_id: str, photo_path: Path, caption: str) -> bool:
        """Envia uma única foto com legenda formatada em HTML."""
        try:
            with open(photo_path, "rb") as f:
                files = {"photo": (photo_path.name, f, "image/jpeg")}
                data = {"chat_id": chat_id, "caption": caption, "parse_mode": "HTML"}
                resp = await self.client.post(f"{self.base_url}/sendPhoto", data=data, files=files)
                return resp.status_code == 200
        except Exception as err:
            logger.debug(f"Erro ao enviar foto para o Telegram: {err}")
            return False

    async def _send_media_group(self, chat_id: str, crop_path: Path, full_path: Path, caption: str) -> bool:
        """Envia um álbum agrupado com a cena ampla e o recorte de close."""
        import json
        try:
            media = [
                {"type": "photo", "media": "attach://crop", "caption": caption, "parse_mode": "HTML"},
                {"type": "photo", "media": "attach://full"}
            ]
            with open(crop_path, "rb") as f_crop, open(full_path, "rb") as f_full:
                files = {
                    "crop": (crop_path.name, f_crop, "image/jpeg"),
                    "full": (full_path.name, f_full, "image/jpeg")
                }
                data = {"chat_id": chat_id, "media": json.dumps(media)}
                resp = await self.client.post(f"{self.base_url}/sendMediaGroup", data=data, files=files)
                return resp.status_code == 200
        except Exception as err:
            logger.debug(f"Falha ao enviar media group Telegram, tentando foto única: {err}")
            return await self._send_single_photo(chat_id, crop_path, caption)

    async def _send_text_message(self, chat_id: str, text: str) -> bool:
        """Envia mensagem de texto simples formatada em HTML."""
        try:
            data = {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}
            resp = await self.client.post(f"{self.base_url}/sendMessage", json=data)
            return resp.status_code == 200
        except Exception as err:
            logger.debug(f"Erro ao enviar mensagem de texto Telegram: {err}")
            return False

    # =========================================================================
    # PROCESSAMENTO DE COMANDOS DO USUÁRIO (LONG POLLING)
    # =========================================================================

    async def _poll_updates(self) -> None:
        """Loop contínuo de escuta de comandos via long-polling assíncrono."""
        logger.info("👂 Polling de comandos do Telegram iniciado.")
        while self._running:
            try:
                params = {"offset": self._last_update_id + 1, "timeout": 25}
                resp = await self.client.get(f"{self.base_url}/getUpdates", params=params, timeout=35.0)

                if resp.status_code == 200:
                    payload = resp.json()
                    for update in payload.get("result", []):
                        self._last_update_id = update["update_id"]
                        if "message" in update and "text" in update["message"]:
                            await self._handle_message(update["message"])

            except asyncio.CancelledError:
                break
            except Exception as e:
                # Evita sobrecarga de logs em caso de interrupção de rede
                logger.debug(f"Polling do Telegram aguardando reconexão: {e}")
                await asyncio.sleep(4.0)

    async def _handle_message(self, msg: Dict[str, Any]) -> None:
        """Processa mensagens e comandos enviados pelos usuários."""
        chat_id = str(msg["chat"]["id"])
        text = msg.get("text", "").strip()
        user_name = msg.get("from", {}).get("first_name", "Usuário")

        configured_chat = settings.telegram_chat_id.strip()

        # Se o TELEGRAM_CHAT_ID ainda não foi configurado, informa o Chat ID no primeiro /start
        if not configured_chat:
            welcome = (
                f"👋 Olá, {user_name}!\n\n"
                f"Seu <b>Chat ID</b> é: <code>{chat_id}</code>\n\n"
                f"👉 Para autorizar este chat a receber os alertas de segurança, adicione no seu arquivo <code>.env</code>:\n"
                f"<code>TELEGRAM_CHAT_ID={chat_id}</code>\n"
                f"E reinicie o serviço com <code>sudo systemctl restart cerberus</code>."
            )
            await self._send_text_message(chat_id, welcome)
            return

        # Segurança: Rejeita comandos de chats não autorizados
        if chat_id != configured_chat:
            logger.warning(f"🚫 Comando não autorizado recebido no Telegram do chat {chat_id} ({user_name})")
            await self._send_text_message(chat_id, "⛔ <b>Acesso Negado.</b> Este bot é privado do sistema Cerberus.")
            return

        cmd = text.split()[0].lower() if text else ""
        # Remove menção do bot caso chamada via grupo (ex: /status@CerberusBot)
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
        """Envia mensagem de boas-vindas e catálogo de comandos disponíveis."""
        help_text = (
            f"🛡️ <b>CERBERUS - SENTINELA VISION</b>\n"
            f"<i>Sistema Autônomo de Vigilância & Visão Computacional</i>\n\n"
            f"Olá, <b>{user_name}</b>! Estou monitorando sua residência 24/7.\n\n"
            f"📋 <b>Comandos Disponíveis:</b>\n"
            f"🔹 <b>/status</b> - Telemetria do servidor, GPU e FPS das câmeras\n"
            f"🔹 <b>/snapshot</b> - Captura fotos das câmeras ao vivo agora\n"
            f"🔹 <b>/alertas</b> - Lista os últimos 5 eventos registrados\n"
            f"🔹 <b>/ajuda</b> - Exibe esta mensagem de suporte\n\n"
            f"🔔 <i>Os alertas com fotos de intrusão, moradores e veículos são enviados automaticamente aqui em tempo real.</i>"
        )
        avatar_path = Path("assets/bot/cerberus_telegram_avatar.jpg")
        if avatar_path.exists():
            await self._send_single_photo(chat_id, avatar_path, help_text)
        else:
            await self._send_text_message(chat_id, help_text)

    async def _cmd_status(self, chat_id: str) -> None:
        """Envia relatório de telemetria das câmeras, GPU e banco de dados."""
        from app.stream.manager import stream_manager
        from app.storage.database import db

        cams_info = []
        for cam in stream_manager.cameras_config:
            if cam.active:
                fps = stream_manager.inference_fps.get(cam.id, 0.0)
                status_icon = "🟢" if fps > 0 else "🟡"
                cams_info.append(f"  {status_icon} <b>{cam.name}</b>: {fps:.1f} FPS")

        cams_str = "\n".join(cams_info) if cams_info else "  Nenhuma câmera ativa."

        # Informações de hardware
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
            f"🕒 <b>Hora do Servidor:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n"
            f"✅ <b>Status:</b> Operação contínua 24/7 sem interrupções."
        )
        await self._send_text_message(chat_id, status_msg)

    async def _cmd_snapshot(self, chat_id: str) -> None:
        """Captura e envia fotos ao vivo de todas as câmeras ativas."""
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
        """Consulta os últimos 5 eventos no SQLite e exibe no chat."""
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


# Instância global singleton do bot (desacoplada)
telegram_bot = TelegramBot()
