import os
from pymongo import MongoClient
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", None)
MONGO_DB = os.getenv("MONGO_DB", None)
ENV = os.getenv("ENV", "prod").lower()

# --- Funções de Coleções ---

def get_mongo_collection(collection_name: str):
    if MONGO_URI is None or MONGO_DB is None:
        raise ValueError("MONGO_URI and MONGO_DB must be set")
    
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    return db[collection_name]


# --- Funções de Log de Previsão ---

def log_prediction(prediction_data: dict) -> dict:
    """
    Insere um log de predição no banco de dados e retorna o
    documento inserido com o ID formatado.
    """
    collection = get_mongo_collection(f"{ENV.upper()}_intent_logs")
    
    # Log the prediction to the database
    try:
        collection.insert_one(prediction_data)
        # If insert_one succeeds, it adds '_id' to the 'results' dict
        prediction_data['id'] = str(prediction_data.get('_id', None))
        prediction_data.pop('_id', None)
    except Exception as e:
        # If insert_one fails, log the error and continue
        raise Exception(f"Failed to log prediction to database. Error: {e}")

    return prediction_data

