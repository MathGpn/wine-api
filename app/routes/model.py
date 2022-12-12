from fastapi.routing import APIRouter
from fastapi import HTTPException
from app.core.config import get_api_settings
from app.classes.models import MLModel, DataLine, ResponseJson
from app.scripts.model_tools import get_metrics, add_data_to_dataset, launch_model_fitting
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


settings = get_api_settings()
API_MODEL_ROUTE = settings.api_model_route
MODEL_FILE = settings.model_file

ModelRouter = APIRouter()

@ModelRouter.get(f"{API_MODEL_ROUTE}", response_model=ResponseJson)
async def get_model_serialized() -> ResponseJson:
    """Give the serialized model
    
    Raises:
        HTTPException: 400 status code if the user input is incorrect
        HTTPException: 500 status code if an error is raised during the process
    
    Returns:
        (ResponseJson): Information about the process
    """
    return {"message": "You have obtained the serialized model in pickle format !", "status_code": 200}

@ModelRouter.get(f"{API_MODEL_ROUTE}/description", response_model=MLModel)
async def get_model_info() -> MLModel:
    """Give technical information about the Machine Learning Model used in this API 
    
    Raises:
        HTTPException: 500 status code if an error is raised during the process
    
    Returns:
        (MLModel): Information About the Machine Learning Model used
    """
    res = {}
    try:
        if not os.path.exists(MODEL_FILE):
            await train_model()
        model = pickle.load(open(MODEL_FILE, "rb"))
        params = model.get_params(deep=True)
        res["model_name"] = model.__class__.__name__
        res["metrics"] = await get_metrics()
        res["training_params"] = params
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the model loading : {e}")
    return res


@ModelRouter.put(API_MODEL_ROUTE, response_model=ResponseJson)
async def add_new_data(data: DataLine) -> ResponseJson:
    """Take user new data and add them to the training dataset

    Args:
        data (DataLine): The inputs given by user

    Raises:
        HTTPException: 400 status code if the user input is incorrect
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (ResponseJson): Information about the process
    """        
    
    try:
        await add_data_to_dataset(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error while adding data : {e}")
    return {"message": "Successfully added data !", "status_code": 200}


@ModelRouter.post(f"{API_MODEL_ROUTE}/retrain", response_model=ResponseJson)
async def train_model() -> ResponseJson:
    """Launch a new fitting of the model with the current dataset 

    Raises:
        HTTPException: 500 status code if an error is raised during the process

    Returns:
        (ResponseJson): Information about the process
    """
    try:
        #model = RandomForestClassifier(criterion= 'entropy', max_depth= 1000, n_estimators= 200)
        model = SVC(random_state=0)
        model = await launch_model_fitting(model)
        pickle.dump(model, open(MODEL_FILE, "wb"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected Error during the model training : {e}")
    return {"message": "The retraining of the model was done successfully !", "status_code": 200}

