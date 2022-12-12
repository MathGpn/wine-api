from app.core.config import get_api_settings
import pandas as pd
from app.classes.models import DataLine, WineQuality
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

settings = get_api_settings()

DATASET_FILE = settings.data_csv
MODEL_TYPE = settings.model_type


async def model_prediction_on_data(model: MODEL_TYPE, data: DataLine)-> WineQuality:
    """ realize a prediction when you are giving in the "body" section all the datas necessary for the wine composition

    Args:
        model (MODEL_TYPE): model from sklearn library already fitted
        data (DataLine): keys : fixed_acidity,volatile_acidity,citric_acid,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol type of values float

    Returns:
        dict: key : predicted_quality, value between 0 and 10
    """
    
    df = pd.read_csv(DATASET_FILE)
    X = df.drop('quality', axis = 1)
    y = df['quality']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    model.fit(X_train, y_train)
    predict_rf = model.predict([[data.fixed_acidity, data.volatile_acidity, data.citric_acid, data.residual_sugar, data.chlorides, data.free_sulfur_dioxide, data.total_sulfur_dioxide, data.density, data.pH, data.sulphates, data.alcohol]])
    classe = WineQuality(predicted_quality=predict_rf)
    return classe

async def generate_best_wine(model : MODEL_TYPE)-> DataLine:
    """ Find the best composition to generate the best wine

    Args:
        model (MODEL_TYPE): model from sklearn library already fitted

    Returns:
        DataLine: keys : fixed_acidity,volatile_acidity,residual_sugar,chlorides,free_sulfur_dioxide,total_sulfur_dioxide,density,pH,sulphates,alcohol type of values float
    """
    df = pd.read_csv(DATASET_FILE)
    best_wines = df[df['quality'] > 7]
    best_wine = DataLine(fixed_acidity = best_wines.iloc[0]['fixed acidity'],
                         volatile_acidity = best_wines.iloc[0]['volatile acidity'],
                         citric_acid = best_wines.iloc[0]['citric acid'],
                         residual_sugar = best_wines.iloc[0]['residual sugar'],
                         chlorides = best_wines.iloc[0]['chlorides'],
                         free_sulfur_dioxide = best_wines.iloc[0]['free sulfur dioxide'],
                         total_sulfur_dioxide = best_wines.iloc[0]['total sulfur dioxide'],
                         density = best_wines.iloc[0]['density'],
                         pH = best_wines.iloc[0]['pH'],
                         sulphates = best_wines.iloc[0]['sulphates'],
                         alcohol = best_wines.iloc[0]['alcohol'],
                         quality = best_wines.iloc[0]['quality'] + 2
                         )
    return best_wine