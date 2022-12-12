from fastapi.routing import APIRouter
from fastapi import HTTPException
from app.routes.model import train_model
from app.core.config import get_api_settings
from app.classes.models import DataLine, WineQuality
from app.scripts.predict_tools import model_prediction_on_data, generate_best_wine
import pickle, os

settings = get_api_settings()
API_PREDICT_ROUTE = settings.api_predict_route
MODEL_FILE = settings.model_file

PredictRouter = APIRouter()

@PredictRouter.post(API_PREDICT_ROUTE, response_model=WineQuality)
async def predict_on_data(data: DataLine) -> WineQuality:
    """Launch a prediction on the giving data 

    Raises:
        HTTPException: 400 status code if the user input is incorrect
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (WineProba): The predicted quality for the wine 
    """
    try:
        if not os.path.exists(MODEL_FILE):
            await train_model()
        model = pickle.load(open(MODEL_FILE, "rb"))
        preds = await model_prediction_on_data(model, data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the model prediction : {e}")
    return preds

@PredictRouter.get(API_PREDICT_ROUTE, response_model=DataLine)
async def get_best_wine() -> DataLine:
    """Generate the description of the best features for the "perfect" wine 

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (DataLine): The wine generated
    """
    try:
        if not os.path.exists(MODEL_FILE):
            await train_model()
        model = pickle.load(open(MODEL_FILE, "rb"))
        draw = await generate_best_wine(model)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the wine generation : {e}")
    return draw
