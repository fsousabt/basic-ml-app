from typing import Dict
from datetime import datetime, timezone
from intent_classifier import IntentClassifier
from db.engine import log_prediction
from app.schema import SinglePrediction, PredictionResponse
import os

# services.py
def load_all_models() -> dict:
    MODELS = {}
    try:
        model_files = [f for f in os.listdir(os.path.join(os.path.dirname(__file__), "..", "intent_classifier", "models")) if f.endswith(".keras")]
        for model_file in model_files:
            model_path = os.path.join(os.path.dirname(__file__), "..", "intent_classifier", "models", model_file)
            model_name = model_file.replace(".keras", "")
            MODELS[model_name] = IntentClassifier(load_model=model_path)
        return MODELS
    except Exception as e:
        raise Exception(f"Failed to load models: {str(e)}")


def predict_and_log_intent(
    text: str, 
    owner: str, 
    models: Dict[str, IntentClassifier]
) -> Dict:
    """
    1. Executa as predições de ML.
    2. Formata o resultado.
    3. Envia o resultado para o log no banco de dados.
    4. Retorna o resultado final formatado.
    """
    
    # 1. Executa predições (Lógica de ML)
    predictions = {}
    for model_name, model in models.items():
        top_intent, all_probs = model.predict(text)
        predictions[model_name] = SinglePrediction(top_intent=top_intent, all_probs=all_probs)

    # 2. Formata o documento de log (Lógica de Dados)
    log_document = PredictionResponse(text=text, 
                                      owner=owner, 
                                      predictions=predictions, 
                                      timestamp=int(datetime.now(timezone.utc).timestamp()))
    
    # 3. Salva no BD (Lógica de Persistência)
    # Convert Pydantic model to dict for MongoDB insertion
    log_dict = log_document.model_dump(exclude_none=True)  # exclude_none to skip id=None
    final_result = log_prediction(log_dict)
    
    # 4. Retorna o resultado
    return final_result


