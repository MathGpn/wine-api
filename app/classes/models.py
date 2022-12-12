from typing import List, Optional
from pydantic import BaseModel, Field    

class WineQuality(BaseModel):
    """The predicted mark  
    """
    predicted_quality: int
    
class ResponseJson(BaseModel):
    """Default Response
    """
    message: str = "OK"
    status_code: int = 200

class Metric(BaseModel):
    """Metric Object containing information about a metric
    """
    metric_name: str
    value: float

    
class MLModel(BaseModel):
    """Machine Learning Model Object
    """
    metrics: List[Metric]
    model_name: str
    training_params : dict
    

class DataLine(BaseModel):
    """Line Data Object with the appropriate dataset format
    """
    fixed_acidity: float = Field(..., example=7.4, gt=0, lt=18)
    volatile_acidity: float = Field(..., example=0.8,gt=0, lt=2)
    citric_acid: float = Field(..., example=0.6,gt=0, lt=1)
    residual_sugar: float = Field(..., example=5.4,gt=0, lt=20)
    chlorides: float = Field(..., example=0.4,gt=0, lt=1)
    free_sulfur_dioxide: float = Field(..., example=17.4,gt=0, lt=100)
    total_sulfur_dioxide: float = Field(..., example=412,gt=0, lt=500)
    density: float = Field(..., example=1,gt=0.7, lt=1.2)
    pH: float = Field(..., example=8.3,gt=0, lt=14)
    sulphates: float = Field(...,example=2, gt=0, lt=3)
    alcohol: float = Field(..., example=9.4, gt=0, lt=20)
    quality: Optional[int] = Field(None, title="The mark for this wine", ge=0)

