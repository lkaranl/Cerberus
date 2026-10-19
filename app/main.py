from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.core.config import settings
from app.core.logging import logger
from app.storage.database import db
from app.detector.engine import detector_engine
from app.stream.manager import stream_manager
from app.api.routes import router
from app.api.websocket import ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info("Iniciando Sentinela Vision...")
    settings.ensure_directories()
    db.init_db()

    # Inicializa engine de IA antecipadamente
    try:
        detector_engine.initialize()
    except Exception as e:
        logger.warning(f"Engine de IA não inicializada imediatamente: {e}. Será carregada na primeira inferência.")

    # Inicia captura de câmeras e loop de inferência
    stream_manager.start()
    logger.info(f"Sentinela Vision pronto e operando na porta {settings.port}")

    yield

    # --- Shutdown ---
    logger.info("Encerrando Sentinela Vision...")
    stream_manager.stop()
    logger.info("Sentinela Vision finalizado com sucesso.")


app = FastAPI(
    title="Sentinela Vision",
    description="Sistema autônomo de monitoramento de segurança com detecção e rastreamento de pessoas por IA.",
    version="1.0.0",
    lifespan=lifespan
)

# Configuração de CORS para permitir acesso local e de rede interna
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Registro das rotas REST e WebSockets
app.include_router(router)
app.include_router(ws_router)

# Monta arquivos estáticos (CSS, JS, assets)
static_path = Path(__file__).parent / "static"
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")


@app.get("/", include_in_schema=False)
async def serve_index():
    """Entrega a interface web de monitoramento."""
    index_file = static_path / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Sentinela Vision API online. Interface estática não encontrada."}
