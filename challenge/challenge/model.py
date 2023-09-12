import pickle
from datetime import datetime

import joblib
import pandas as pd
import numpy as np

#from AuxFunc import get_rate_from_column, get_min_diff, get_period_day, is_high_season
from typing import Tuple, Union, List
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

model_name = "../challenge/reg_model.pkl"
threshold_in_minutes = 15
MODEL = joblib.load('reg_model.sav')

class DelayModel:

    def __init__(
        self
    ):
        self._model = MODEL # Model should be saved in this attribute.

    def preprocess(
        self,
        data: pd.DataFrame,
        target_column: str = None
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], pd.DataFrame]:
        """
        Prepare raw data for training or predict.

        Args:
            data (pd.DataFrame): raw data.
            target_column (str, optional): if set, the target is returned.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: features and target.
            or
            pd.DataFrame: features.
        """
        # period day of the flights date
        data['period_day'] = data['Fecha-I'].apply(get_period_day)
        data['high_season'] = data['Fecha-I'].apply(is_high_season)
        data['min_diff'] = data.apply(get_min_diff, axis=1)
        if target_column:
            data[target_column] = np.where(data['min_diff'] > threshold_in_minutes, 1, 0)
            target = data[target_column]
        features = pd.concat([
            pd.get_dummies(data['OPERA'], prefix='OPERA'),
            pd.get_dummies(data['TIPOVUELO'], prefix='TIPOVUELO'),
            pd.get_dummies(data['MES'], prefix='MES')],
            axis=1
        )
        return Union[Tuple[features, target], features]

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.DataFrame
    ) -> None:
        """
        Fit model with preprocessed data.

        Args:
            features (pd.DataFrame): preprocessed data.
            target (pd.DataFrame): target.
        """
        x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.33, random_state=42)
        self._model.fit(x_train, y_train)
        return self

    def predict(
        self,
        features: pd.DataFrame
    ) -> List[int]:
        """
        Predict delays for new flights.

        Args:
            features (pd.DataFrame): preprocessed data.
        
        Returns:
            (List[int]): predicted targets.
        """
        y_predicts = self._model.predict(features)
        return y_predicts

# Auxilary functions
def get_period_day(date):
    """
    State the period of the day for each date
    :param date: Flight date
    :return: string - period (morning, afternoon, night)
    """
    date_time = datetime.strptime(date, '%Y-%m-%d %H:%M:%S').time()
    morning_min = datetime.strptime("05:00", '%H:%M').time()
    morning_max = datetime.strptime("11:59", '%H:%M').time()
    afternoon_min = datetime.strptime("12:00", '%H:%M').time()
    afternoon_max = datetime.strptime("18:59", '%H:%M').time()
    evening_min = datetime.strptime("19:00", '%H:%M').time()
    evening_max = datetime.strptime("23:59", '%H:%M').time()
    night_min = datetime.strptime("00:00", '%H:%M').time()
    night_max = datetime.strptime("4:59", '%H:%M').time()

    if (date_time > morning_min and date_time < morning_max):
        return 'mañana'
    elif (date_time > afternoon_min and date_time < afternoon_max):
        return 'tarde'
    elif (
            (date_time > evening_min and date_time < evening_max) or
            (date_time > night_min and date_time < night_max)
    ):
        return 'noche'
def is_high_season(fecha):
    """
    Determine if the flight date is in a high season or not
    :param fecha: Flight date
    return: yes =1 no=0
    """
    fecha_año = int(fecha.split('-')[0])
    fecha = datetime.strptime(fecha, '%Y-%m-%d %H:%M:%S')
    range1_min = datetime.strptime('15-Dec', '%d-%b').replace(year=fecha_año)
    range1_max = datetime.strptime('31-Dec', '%d-%b').replace(year=fecha_año)
    range2_min = datetime.strptime('1-Jan', '%d-%b').replace(year=fecha_año)
    range2_max = datetime.strptime('3-Mar', '%d-%b').replace(year=fecha_año)
    range3_min = datetime.strptime('15-Jul', '%d-%b').replace(year=fecha_año)
    range3_max = datetime.strptime('31-Jul', '%d-%b').replace(year=fecha_año)
    range4_min = datetime.strptime('11-Sep', '%d-%b').replace(year=fecha_año)
    range4_max = datetime.strptime('30-Sep', '%d-%b').replace(year=fecha_año)

    if ((fecha >= range1_min and fecha <= range1_max) or
            (fecha >= range2_min and fecha <= range2_max) or
            (fecha >= range3_min and fecha <= range3_max) or
            (fecha >= range4_min and fecha <= range4_max)):
        return 1
    else:
        return 0

def get_min_diff(data):
    """
    Calculates the difference in minute of the delayed flight
    :param data: Flight date
    :return: float - min_diff difference in minuts
    """
    fecha_o = datetime.strptime(data['Fecha-O'], '%Y-%m-%d %H:%M:%S')
    fecha_i = datetime.strptime(data['Fecha-I'], '%Y-%m-%d %H:%M:%S')
    min_diff = ((fecha_o - fecha_i).total_seconds()) / 60
    return min_diff


def get_rate_from_column(data, column):
    """
    Calculates the rate of delay for a specific column
    :param data: DF data from the flights
    :param column: column to be analyzed
    :return: DF with rate of delays
    """
    delays = {}
    for _, row in data.iterrows():
        if row['delay'] == 1:
            if row[column] not in delays:
                delays[row[column]] = 1
            else:
                delays[row[column]] += 1
    total = data[column].value_counts().to_dict()

    rates = {}
    for name, total in total.items():
        if name in delays:
            rates[name] = round(total / delays[name], 2)
        else:
            rates[name] = 0

    return pd.DataFrame.from_dict(data=rates, orient='index', columns=['Tasa (%)'])