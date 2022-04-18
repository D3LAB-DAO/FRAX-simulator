import warnings  # nopep8
warnings.filterwarnings('ignore')  # nopep8

from datetime import datetime
import pandas as pd

from darts.dataprocessing.transformers import Scaler


def read_csv_with_ffill(path: str, start_date: datetime = None, default_values: dict = None):
    """_summary_

    Args:
        path (str): _description_
        start_date (datetime, optional): _description_. Defaults to None.
        default_values (dict, optional): _description_. Defaults to None.

    Returns:
        df (pd.DataFrame): includes 'price' and 'total_volume'.
    """

    df = pd.read_csv(path, delimiter=",", parse_dates=['snapped_at'])
    df['snapped_at'] = df['snapped_at'].dt.date
    df['snapped_at'] = pd.to_datetime(df['snapped_at']).dt.tz_localize(None)  # Remove Timezone

    df.set_index('snapped_at', inplace=True)
    df = df[start_date:]
    if df.index[0] > start_date:
        new_df = pd.DataFrame([default_values], index=[start_date])
        new_df.index.name = 'snapped_at'
        df = df.append(new_df)
    df = df.resample('D').ffill().reset_index()  # forward fill

    return df


def get_scaler_transformer(data):
    transformer = Scaler()
    transformer.fit_transform(data)
    return transformer
