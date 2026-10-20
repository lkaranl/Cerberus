#!/usr/bin/env python3
"""
Cerberus - Teste & Diagnóstico do Módulo Telegram
Verifica conectividade com a API do Telegram, autenticação do Bot e envio de alerta de teste.
"""

import sys
import asyncio
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.core.config import settings
from app.bot.telegram_bot import telegram_bot

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"   {title}")
    print("=" * 70)

async def test_telegram():
    print_header("Cerberus - Teste do Bot & Notificador Telegram")

    print(f"• TELEGRAM_ENABLED: {settings.telegram_enabled}")
    token_masked = (settings.telegram_bot_token[:6] + "..." + settings.telegram_bot_token[-4:]) if len(settings.telegram_bot_token) > 10 else "(não configurado)"
    print(f"• TELEGRAM_BOT_TOKEN: {token_masked}")
    print(f"• TELEGRAM_CHAT_ID: {settings.telegram_chat_id or '(não configurado)'}")

    if not settings.telegram_enabled or not settings.telegram_bot_token:
        print("\n⚠️ O Módulo Telegram está atualmente desativado ou sem token.")
        print("\n📖 COMO CONFIGURAR EM 1 MINUTO:")
        print("  1. Abra o Telegram e converse com o bot oficial @BotFather")
        print("  2. Envie o comando /newbot e siga as instruções para criar seu bot")
        print("  3. Copie o TOKEN gerado (ex: 7123456789:AAH...)")
        print("  4. Adicione no arquivo .env do servidor:")
        print("       TELEGRAM_ENABLED=true")
        print("       TELEGRAM_BOT_TOKEN=seu_token_aqui")
        print("       TELEGRAM_CHAT_ID=seu_chat_id")
        print("  5. (Dica para descobrir seu Chat ID: inicie uma conversa com o bot recém-criado e envie /start)")
        print("\n======================================================================\n")
        return

    print("\n⏳ Conectando e validando token com a API do Telegram...")
    started = await telegram_bot.start()
    if not started:
        print("❌ Falha na conexão ou token inválido. Verifique o TELEGRAM_BOT_TOKEN no .env.")
        return

    print(f"✅ Bot Autenticado com Sucesso: @{telegram_bot.bot_username}")

    if not settings.telegram_chat_id:
        print("⚠️ TELEGRAM_CHAT_ID ainda não está preenchido no .env.")
        print(f"👉 Abra o Telegram, pesquise por @{telegram_bot.bot_username} e envie qualquer mensagem.")
        print("   O bot responderá imediatamente informando o seu Chat ID numérico!")
        await telegram_bot.stop()
        return

    print(f"🚀 Enviando mensagem e foto de teste para o Chat ID: {settings.telegram_chat_id}...")

    avatar_path = Path("assets/bot/cerberus_telegram_avatar.jpg")
    test_caption = (
        "🤖 <b>CERBERUS SENTINEL - TESTE DE INTEGRAÇÃO</b>\n\n"
        "✅ O módulo Telegram foi inicializado e integrado com sucesso!\n"
        "🔔 Este canal agora receberá fotos dos alertas de segurança em tempo real.\n\n"
        "Comandos disponíveis: <b>/status</b> | <b>/snapshot</b> | <b>/alertas</b>"
    )

    if avatar_path.exists():
        ok = await telegram_bot._send_single_photo(settings.telegram_chat_id, avatar_path, test_caption)
    else:
        ok = await telegram_bot._send_text_message(settings.telegram_chat_id, test_caption)

    if ok:
        print("✅ Mensagem de teste enviada com sucesso! Verifique seu Telegram.")
    else:
        print("❌ Não foi possível entregar a mensagem. Verifique se você já deu /start no bot pelo menos uma vez.")

    await telegram_bot.stop()
    print("\n" + "=" * 70 + "\n")

if __name__ == "__main__":
    asyncio.run(test_telegram())
