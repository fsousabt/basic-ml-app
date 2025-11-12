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
from db.engine import get_mongo_collection
from app.auth import verify_token

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
        "http://localhost:3000",  # React ou outra frontend local
        "https://meusite.com",    # domínio em produção
    ],
    allow_credentials=True,
    allow_methods=["*"],              # permite todos os métodos: GET, POST, etc
    allow_headers=["*"],              # permite todos os headers (Authorization, Content-Type...)
    # Durante o desenvolvimento: você pode usar allow_origins=["*"] para liberar tudo.
    # Em produção: evite "*" e especifique os domínios confiáveis.
)

# Initialize database connection
collection = None
try:
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    logger.info("Database connection established")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    logger.error(traceback.format_exc())


async def conditional_auth():
    """Returns user based on environment mode"""
    global ENV
    if ENV == "dev":
        logger.info("Development mode: skipping authentication")
        return "dev_user"
    else:
        try:
            return verify_token() 
        except HTTPException as he:
            raise he
        except Exception as e:
            # 3. Catch any *other* unexpected errors
            logger.error(f"Unexpected authentication error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=401, detail="Authentication failed")


# Load models
MODELS = {}
try:
    logger.info("Loading confusion model...")
    # Load all the .keras files in the intent_classifier/models folder
    model_files = [f for f in os.listdir(os.path.join(os.path.dirname(__file__), "..", "intent_classifier", "models")) if f.endswith(".keras")]
    for model_file in model_files:
        model_path = os.path.join(os.path.dirname(__file__), "..", "intent_classifier", "models", model_file)
        model_name = model_file.replace(".keras", "")
        MODELS[model_name] = IntentClassifier(load_model=model_path)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    logger.error(traceback.format_exc())


"""
Routes
"""

@app.get("/")
async def root():
    return {"message": f"Basic ML App is running in {ENV} mode"}


@app.post("/predict")
async def predict(text: str, owner: str = Depends(conditional_auth)):
    # ... (prediction generation code is the same) ...
    predictions = {}
    for model_name, model in MODELS.items():
        top_intent, all_probs = model.predict(text)
        predictions[model_name] = {
            "top_intent": top_intent,
            "all_probs": all_probs
        }

    results = {
        "text": text, 
        "owner": owner, 
        "predictions": predictions, 
        "timestamp": int(datetime.now(timezone.utc).timestamp())
    }
    
    # Log the prediction to the database
    try:
        collection.insert_one(results)
        # If insert_one succeeds, it adds '_id' to the 'results' dict
        results['id'] = str(results.get('_id'))
        results.pop('_id', None)
    except Exception as e:
        # If insert_one fails, log the error and continue
        logger.error(f"CRITICAL: Failed to log prediction to database. Error: {e}")
        logger.error(traceback.format_exc())
        # We can set 'id' to None to show it wasn't logged
        results['id'] = None
        # Make sure to pop '_id' in case insert_one added it but failed after
        results.pop('_id', None)

    return JSONResponse(content=results)



if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)