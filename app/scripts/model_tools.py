from app.core.config import get_api_settings
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
import csv, json
from typing import Dict, List
import pandas as pd
from app.classes.models import DataLine, Metric

settings = get_api_settings()

DATASET_FILE = settings.data_csv
METRICS_FILE = settings.metrics_json
MODEL_TYPE = settings.model_type

async def get_metrics()->List[Metric]:
    """ Read json file and format data to match with List[Metric] type

    Returns:
        List[Metric]: list of metric for the current model app
    """
    metrics: List[Metric] = []
    with open(METRICS_FILE, 'r') as f:
        data = json.load(f)
    for metric_name, value in data.items():
           metrics.append(Metric(metric_name=metric_name, value=value))
    return metrics

async def set_metrics(model: MODEL_TYPE, X_test: pd.DataFrame, y_test: pd.DataFrame)->None:
    """ Calcul and save in a json file differents metrics of the model

    Args:
        model (MODEL_TYPE): model from sklearn library
        X_test (pd.DataFrame): values for testing the model
        y_test (pd.DataFrame): output of the testing value

    Returns:
        [type]: Nothing
    """
    metrics: Dict[str,float] = {}
    y_pred = model.predict(X_test)
    recall_value = recall_score(y_test, y_pred, average="weighted")
    metrics["Recall"] = recall_value
    precision_value = precision_score(y_test, y_pred, average="weighted")
    metrics["Precision"] = precision_value
    f1_value = f1_score(y_test, y_pred, average="weighted")
    metrics["F1"] = f1_value
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f)
    return None

async def add_data_to_dataset(data: DataLine)->None:
    """ Add a new row at the end of the Wines.csv

    Args:
        data (dict): keys : fixed_acidity,volatile_acidity,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol values float
                     key : quality value int 

    Returns:
        [type]: Nothing
    """

    with open(DATASET_FILE, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([data.fixed_acidity,data.volatile_acidity,data.citric_acid,data.residual_sugar,data.chlorides,data.free_sulfur_dioxide,data.total_sulfur_dioxide,data.density,data.pH,data.sulphates,data.alcohol, data.quality])
    return None

async def launch_model_fitting(model: MODEL_TYPE)->MODEL_TYPE:
    """ preprocess data and fit model variable

    Args:
        model (MODEL_TYPE): model from sklearn library

    Returns:
        MODEL_TYPE: model from sklearn library fitted
    """

    df = pd.read_csv(DATASET_FILE)
    X = df.drop('quality', axis = 1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.9,random_state=0)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model.fit(X_train,y_train)
    await set_metrics(model, X_test, y_test)
        
    return model