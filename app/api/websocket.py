import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from app.stream.manager import stream_manager
from app.core.logging import logger

ws_router = APIRouter(prefix="/ws", tags=["WebSockets"])


@ws_router.websocket("/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: str):
    """
    Streaming contínuo de frames JPEG processados com bounding boxes em tempo real.
    Envia frames binários otimizados diretamente ao cliente para renderização fluida no Canvas.
    """
    await websocket.accept()
    logger.info(f"Cliente WebSocket conectado ao stream da câmera: {camera_id}")

    try:
        last_sent_bytes = None
        while True:
            jpeg_bytes = stream_manager.latest_annotated_jpeg.get(camera_id)

            if jpeg_bytes is not None and jpeg_bytes is not last_sent_bytes:
                await websocket.send_bytes(jpeg_bytes)
                last_sent_bytes = jpeg_bytes

            # Taxa de envio de até 25-30 FPS dependendo da captura
            await asyncio.sleep(0.033)

    except WebSocketDisconnect:
        logger.info(f"Cliente desconectou do stream da câmera: {camera_id}")
    except Exception as e:
        logger.warning(f"Erro na conexão WebSocket ({camera_id}): {e}")
        try:
            await websocket.close()
        except Exception:
            pass


@ws_router.websocket("/events")
async def websocket_events(websocket: WebSocket):
    """
    Canal de eventos em tempo real.
    Dispara notificações instantâneas com dados da pessoa detectada e URLs das evidências.
    """
    await websocket.accept()
    queue = stream_manager.register_event_listener()
    logger.info("Cliente conectado ao canal de eventos em tempo real.")

    try:
        while True:
            # Aguarda próximo evento disparado pelo tracker
            event = await queue.get()
            await websocket.send_text(json.dumps({
                "type": "PERSON_DETECTED",
                "data": event
            }))
    except WebSocketDisconnect:
        logger.info("Cliente desconectou do canal de eventos.")
    except Exception as e:
        logger.warning(f"Erro no WebSocket de eventos: {e}")
    finally:
        stream_manager.unregister_event_listener(queue)
        try:
            await websocket.close()
        except Exception:
            pass
