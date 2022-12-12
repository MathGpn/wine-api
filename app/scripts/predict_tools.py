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
    mean_fixed_acidity = 0
    mean_volatile_acidity = 0
    mean_alcohol = 0
    mean_chlorides = 0 
    mean_citric_acid = 0
    mean_density = 0
    mean_pH = 0
    mean_residual_sugar = 0
    mean_sulphates = 0
    mean_free_sulfur_dioxide = 0
    mean_total_sulfur_dioxide = 0
    
    best_wines = df[df['quality'] > 7]
    for i in range (len(best_wines)) :
        mean_fixed_acidity += best_wines.iloc[i][0] 
        mean_volatile_acidity += best_wines.iloc[i][1]
        mean_citric_acid += best_wines.iloc[i][2]
        mean_residual_sugar += best_wines.iloc[i][3]
        mean_chlorides += best_wines.iloc[i][4]
        mean_free_sulfur_dioxide += best_wines.iloc[i][5]
        mean_total_sulfur_dioxide += best_wines.iloc[i][6]
        mean_density += best_wines.iloc[i][7]
        mean_pH += best_wines.iloc[i][8]
        mean_sulphates += best_wines.iloc[i][9]
        mean_alcohol += best_wines.iloc[i][10]

    mean_fixed_acidity = mean_fixed_acidity / len(best_wines)
    mean_volatile_acidity = mean_volatile_acidity / len(best_wines)
    mean_citric_acid = mean_citric_acid / len(best_wines)
    mean_residual_sugar = mean_residual_sugar / len(best_wines)
    mean_chlorides = mean_chlorides / len(best_wines)
    mean_free_sulfur_dioxide = mean_free_sulfur_dioxide / len(best_wines)
    mean_total_sulfur_dioxide = mean_total_sulfur_dioxide / len(best_wines)
    mean_density = mean_density / len(best_wines)
    mean_pH = mean_pH / len(best_wines)
    mean_sulphates = mean_sulphates / len(best_wines)
    mean_alcohol = mean_alcohol / len(best_wines)
    mean_quality = 8

    best_wine = DataLine(   
                            fixed_acidity = mean_fixed_acidity,
                            volatile_acidity = mean_volatile_acidity,
                            citric_acid = mean_citric_acid,
                            residual_sugar = mean_residual_sugar,
                            chlorides = mean_chlorides,
                            free_sulfur_dioxide = mean_free_sulfur_dioxide,
                            total_sulfur_dioxide = mean_total_sulfur_dioxide,
                            density = mean_density,
                            pH = mean_pH,
                            sulphates = mean_sulphates,
                            alcohol = mean_alcohol,
                            quality = mean_quality + 2
                        )
    return best_wine