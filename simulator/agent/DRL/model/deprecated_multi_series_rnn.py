from datetime import datetime
from tabnanny import verbose
from tracemalloc import start
import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler
from darts.utils.likelihood_models import GaussianLikelihood
from darts.metrics import mape


def read_csv_with_ffill(path: str, start_date: datetime = None, default_values: dict = None):
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


"""Hyper Params"""
N_EPOCHS = 300  # 300
FORCE_RESET = True

N_VAL = 21
N_PRED = 360

"""Load Data"""
df_btc = read_csv_with_ffill('./data/btc-usd-max.csv', datetime(2019, 5, 8))  # BTC-USD
df_luna = read_csv_with_ffill('./data/luna-usd-max.csv', datetime(2019, 5, 8))  # LUNA-USD
df_ust = read_csv_with_ffill('./data/ust-usd-max.csv', datetime(2019, 5, 8), default_values={
    'price': 1.0,
    'market_cap': None,
    'total_volume': None,
})  # UST-USD

# Create a TimeSeries, specifying the time and value columns
series_btc = TimeSeries.from_dataframe(df_btc, time_col='snapped_at', value_cols='price', freq='D')
series_luna = TimeSeries.from_dataframe(df_luna, time_col='snapped_at', value_cols='price', freq='D')
series_ust = TimeSeries.from_dataframe(df_ust, time_col='snapped_at', value_cols='price', freq='D')

# Set aside the last N_VAL days as a validation series
# train, val = series.split_after(pd.Timestamp("20210101"))
train_btc, val_btc = series_btc[:-N_VAL], series_btc[-N_VAL:]
train_luna, val_luna = series_luna[:-N_VAL], series_luna[-N_VAL:]
train_ust, val_ust = series_ust[:-N_VAL], series_ust[-N_VAL:]

"""covariate"""
# TODO
# 4-months (Quadraple Witching Day)
# 1-week (Weekend/Weekdays)
## 4-year (halving date)
# 1-year (annual financial plan)

"""Normalize"""
# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer_btc = Scaler()
transformer_luna = Scaler()
transformer_ust = Scaler()

train_btc_transformed = transformer_btc.fit_transform(train_btc)  # Fit at Training datasaet
val_btc_transformed = transformer_btc.transform(val_btc)
series_btc_transformed = transformer_btc.transform(series_btc)

train_luna_transformed = transformer_luna.fit_transform(train_luna)  # Fit at Training datasaet
val_luna_transformed = transformer_luna.transform(val_luna)
series_luna_transformed = transformer_luna.transform(series_luna)

train_ust_transformed = transformer_ust.fit_transform(train_ust)  # Fit at Training datasaet
val_ust_transformed = transformer_ust.transform(val_ust)
series_ust_transformed = transformer_ust.transform(series_ust)

"""DL"""
model = BlockRNNModel(
    model="GRU",
    model_name="LUNA-UST_GRU",

    input_chunk_length=7,
    output_chunk_length=1,
    hidden_size=25,
    n_rnn_layers=2,  # 2
    dropout=0.1,  # 0.1

    batch_size=32,
    n_epochs=N_EPOCHS,
    optimizer_kwargs={"lr": 1e-3},
    likelihood=GaussianLikelihood(),  # ignore loss_fn
    # optimizer_cls # use Adam as default
    # pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True},

    random_state=327,
    log_tensorboard=True,
    force_reset=FORCE_RESET,  # at darts_logs/
    save_checkpoints=True,
    show_warnings=False
)
model.fit(
    [train_btc_transformed, train_luna_transformed, train_ust_transformed],
    val_series=[val_btc_transformed, val_luna_transformed, val_ust_transformed],
    verbose=True
)
# model = BlockRNNModel.load_from_checkpoint(model_name="LUNA-UST_GRU")  # , best=True)

"""Eval & Visualization"""
pred_list = model.predict(
    N_PRED,
    series=[train_btc_transformed, train_luna_transformed, train_ust_transformed],
    num_samples=1000,
    # verbose=False
)

for pred_series, label in zip(pred_list, ["BTC", "LUNA", "UST"]):
    if label == "BTC":
        print("Predict: MAPE [{}] = {:.3f}%".format(label, mape(val_btc_transformed, pred_series[:N_VAL])))  # best at 0%
        series_btc.plot(label="actual BTC")
        pred_series_inverse_trasform = transformer_btc.inverse_transform(pred_series)
    elif label == "LUNA":
        print("Predict: MAPE [{}] = {:.3f}%".format(label, mape(val_luna_transformed, pred_series[:N_VAL])))  # best at 0%
        series_luna.plot(label="actual LUNA")
        pred_series_inverse_trasform = transformer_luna.inverse_transform(pred_series)
    elif label == "UST":
        print("Predict: MAPE [{}] = {:.3f}%".format(label, mape(val_ust_transformed, pred_series[:N_VAL])))  # best at 0%
        series_ust.plot(label="actual UST")
        pred_series_inverse_trasform = transformer_ust.inverse_transform(pred_series)
    else:
        raise KeyError("Invalid Ticker")

    pred_series_inverse_trasform.plot(  # pred_series.plot
        label=f"forecast {label}",
        low_quantile=0.05,
        high_quantile=0.95  # Plot the median, 5th and 95th percentiles
    )

    plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.savefig("./figs/{}_pred.png".format(label), dpi=1200)
    plt.close()

"""Backtest"""
backtest_list = list()
for series_transformed in [series_btc_transformed, series_luna_transformed, series_ust_transformed]:
    backtest_list.append(model.historical_forecasts(
        series_transformed,
        num_samples=1000,
        # start=0.6,
        forecast_horizon=1,
        stride=7,
        retrain=False,
        last_points_only=True,
        # verbose=False
    ))

for backtest_series, label in zip(backtest_list, ["BTC", "LUNA", "UST"]):
    if label == "BTC":
        # print("Backtest: MAPE [{}] = {:.3f}%".format(label, mape(series_btc_transformed, backtest_series)))  # best at 0%
        series_btc.plot(label="actual BTC")
        backtest_series_inverse_trasform = transformer_btc.inverse_transform(backtest_series)
    elif label == "LUNA":
        # print("Backtest: MAPE [{}] = {:.3f}%".format(label, mape(series_luna_transformed, backtest_series)))  # best at 0%
        series_luna.plot(label="actual LUNA")
        backtest_series_inverse_trasform = transformer_luna.inverse_transform(backtest_series)
    elif label == "UST":
        # print("Backtest: MAPE [{}] = {:.3f}%".format(label, mape(series_ust_transformed, backtest_series)))  # best at 0%
        series_ust.plot(label="actual UST")
        backtest_series_inverse_trasform = transformer_ust.inverse_transform(backtest_series)
    else:
        raise KeyError("Invalid Ticker")

    backtest_series_inverse_trasform.plot(  # backtest_series.plot
        label=f"forecast {label}",
        low_quantile=0.05,
        high_quantile=0.95  # Plot the median, 5th and 95th percentiles
    )

    plt.legend()
    plt.savefig("./figs/{}_backtest.png".format(label), dpi=1200)
    plt.close()
