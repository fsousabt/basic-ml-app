import os
import re
import traceback
from datetime import datetime
from datetime import timezone
from dotenv import load_dotenv
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from intent_classifier import IntentClassifier
from db.auth import conditional_auth
from app import services


logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Read environment mode (defaults to prod for safety)
ENV = os.getenv("ENV", "prod").lower()
logger.info(f"Running in {ENV} mode")

# Initialize FastAPI app
app = FastAPI(
    title="Basic ML App",
    description="A basic ML app",
    version="1.0.0",
)

# Controle de CORS (Cross-Origin Resource Sharing) para prevenir ataques de fontes não autorizadas.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost",
        # "http://localhost:3000",  # React ou outra frontend local
        # "https://meusite.com",    # domínio em produção
    ],
    allow_credentials=True,
    allow_methods=["*"],              # permite todos os métodos: GET, POST, etc
    allow_headers=["*"],              # permite todos os headers (Authorization, Content-Type...)
    # Durante o desenvolvimento: você pode usar allow_origins=["*"] para liberar tudo.
    # Em produção: evite "*" e especifique os domínios confiáveis.
)


# Initialize models
MODELS = {}
try: 
    logger.info("Loading models...")
    MODELS = services.load_all_models()
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    logger.error(traceback.format_exc())
    raise Exception(f"Failed to load models: {str(e)}")


"""
Routes
"""
@app.get("/")
async def root():
    return {"message": "Basic ML App is running in {ENV} mode"}

@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    """
    Endpoint de predição.
    Este é um 'Controller' enxuto. Ele apenas delega.
    """
    try:
        # 1. O Controller delega TODA a lógica de negócio para o Serviço
        results = services.predict_and_log_intent(
            text=text, 
            owner=owner, 
            models=MODELS
        )
        
        # 2. O Controller retorna a resposta (Lógica de View)
        return JSONResponse(content=results)
        
    except Exception as e:
        logger.error(f"Erro na predição: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Erro interno ao processar a predição.")



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)