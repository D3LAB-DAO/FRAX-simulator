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
from darts.timeseries import concatenate


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
N_EPOCHS = 1000  # 300
FORCE_RESET = True

N_VAL = 21
N_PRED = 360

"""Load Data"""
df_btc = read_csv_with_ffill('./data/btc-usd-max.csv', datetime(2020, 10, 2))  # BTC-USD
df_luna = read_csv_with_ffill('./data/luna-usd-max.csv', datetime(2020, 10, 2))  # LUNA-USD
df_ust = read_csv_with_ffill('./data/ust-usd-max.csv', datetime(2020, 10, 2))  # UST-USD

# Create a TimeSeries, specifying the time and value columns
series_btc = TimeSeries.from_dataframe(df_btc, time_col='snapped_at', value_cols=['price', 'total_volume'], freq='D')
series_luna = TimeSeries.from_dataframe(df_luna, time_col='snapped_at', value_cols=['price', 'total_volume'], freq='D')
series_ust = TimeSeries.from_dataframe(df_ust, time_col='snapped_at', value_cols=['price', 'total_volume'], freq='D')

series_btc = series_btc.with_columns_renamed(['price', 'total_volume'], ['price_btc', 'volume_btc'])
series_luna = series_luna.with_columns_renamed(['price', 'total_volume'], ['price_luna', 'volume_luna'])
series_ust = series_ust.with_columns_renamed(['price', 'total_volume'], ['price_ust', 'volume_ust'])

series_luna_ust = concatenate([series_luna, series_ust], axis=1)

# Set aside the last N_VAL days as a validation series
# train, val = series.split_after(pd.Timestamp("20210101"))
train_btc, val_btc = series_btc[:-N_VAL], series_btc[-N_VAL:]
train_luna_ust, val_luna_ust = series_luna_ust[:-N_VAL], series_luna_ust[-N_VAL:]

"""Normalize"""
# Normalize the time series (note: we avoid fitting the transformer on the validation set)
transformer_btc = Scaler()
transformer_luna_ust = Scaler()

train_btc_transformed = transformer_btc.fit_transform(train_btc)  # Fit at Training datasaet
val_btc_transformed = transformer_btc.transform(val_btc)
series_btc_transformed = transformer_btc.transform(series_btc)

train_luna_ust_transformed = transformer_luna_ust.fit_transform(train_luna_ust)  # Fit at Training datasaet
val_luna_ust_transformed = transformer_luna_ust.transform(val_luna_ust)
series_luna_ust_transformed = transformer_luna_ust.transform(series_luna_ust)

"""DL"""
# BTC
model_btc = BlockRNNModel(
    model="GRU",
    model_name="BTC_GRU",

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
model_btc.fit(
    train_btc_transformed,
    val_series=val_btc_transformed,
    verbose=True
)
# model = BlockRNNModel.load_from_checkpoint(model_name="LUNA-UST_GRU")  # , best=True)

# LUNA-UST
model_luna_ust = BlockRNNModel(
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
model_luna_ust.fit(
    train_luna_ust_transformed,
    past_covariates=train_btc_transformed,
    val_series=val_luna_ust_transformed,
    val_past_covariates=val_btc_transformed,
    verbose=True
)
# model = BlockRNNModel.load_from_checkpoint(model_name="LUNA-UST_GRU")  # , best=True)

"""Eval & Visualization"""
# BTC
pred_series_btc = model_btc.predict(
    N_PRED,
    series=train_btc_transformed,
    num_samples=1,
    # verbose=False
)

# LUNA-UST
pred_series_luna_ust = model_luna_ust.predict(
    N_PRED,
    series=train_luna_ust_transformed,
    past_covariates=concatenate([train_btc_transformed, pred_series_btc[-N_PRED:]], axis=0),
    num_samples=1000,
    # verbose=False
)

print("Predict: MAPE [BTC] = {:.3f}%".format(mape(val_btc_transformed, pred_series_btc[:N_VAL])))  # best at 0%
print("Predict: MAPE [LUNA_UST] = {:.3f}%".format(mape(val_luna_ust_transformed, pred_series_luna_ust[:N_VAL])))  # best at 0%

pred_series_btc_inverse_trasform = transformer_btc.inverse_transform(pred_series_btc)
pred_series_luna_ust_inverse_trasform = transformer_luna_ust.inverse_transform(pred_series_luna_ust)

# BTC
series_btc['price_btc'].plot(label="actual BTC")
pred_series_btc_inverse_trasform['price_btc'].plot(  # pred_series.plot
    label=f"forecast BTC"
)

plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig("./figs/BTC_predict.png", dpi=1200)
plt.close()

# LUNA
series_luna['price_luna'].plot(label="actual LUNA")
pred_series_luna_ust_inverse_trasform['price_luna'].plot(  # pred_series.plot
    label=f"forecast LUNA",
    low_quantile=0.05,
    high_quantile=0.95  # Plot the median, 5th and 95th percentiles
)

plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig("./figs/LUNA_predict.png", dpi=1200)
plt.close()

# UST
series_ust['price_ust'].plot(label="actual UST")
pred_series_luna_ust_inverse_trasform['price_ust'].plot(  # pred_series.plot
    label=f"forecast UST",
    low_quantile=0.05,
    high_quantile=0.95  # Plot the median, 5th and 95th percentiles
)

plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)
plt.legend()
plt.savefig("./figs/UST_predict.png", dpi=1200)
plt.close()

"""Backtest"""
# BTC
backtest_series_btc = model_btc.historical_forecasts(
    series_btc_transformed,
    num_samples=1,
    # start=0.6,
    forecast_horizon=1,
    stride=7,
    retrain=False,
    last_points_only=True,
    # verbose=False
)

# LUNA-UST
backtest_series_luna_ust = model_luna_ust.historical_forecasts(
    series_luna_ust_transformed,
    past_covariates=series_btc_transformed,
    num_samples=1000,
    # start=0.6,
    forecast_horizon=1,
    stride=7,
    retrain=False,
    last_points_only=True,
    # verbose=False
)

print("Backtest: MAPE [BTC] = {:.3f}%".format(mape(series_btc_transformed, backtest_series_btc)))  # best at 0%
print("Backtest: MAPE [LUNA_UST] = {:.3f}%".format(mape(series_luna_ust_transformed, backtest_series_luna_ust)))  # best at 0%

backtest_series_btc_inverse_trasform = transformer_btc.inverse_transform(backtest_series_btc)
backtest_series_btc_inverse_trasform = backtest_series_btc_inverse_trasform.with_columns_renamed(
    ['0', '1'],
    ['price_btc', 'volume_btc']
)

backtest_series_luna_ust_inverse_trasform = transformer_luna_ust.inverse_transform(backtest_series_luna_ust)
backtest_series_luna_ust_inverse_trasform = backtest_series_luna_ust_inverse_trasform.with_columns_renamed(
    ['0', '1', '2', '3'],
    ['price_luna', 'volume_luna', 'price_ust', 'volume_ust']
)

# BTC
series_btc['price_btc'].plot(label="actual BTC")  # price_btc, volume_btc
backtest_series_btc_inverse_trasform['price_btc'].plot(  # backtest_series.plot
    label=f"forecast BTC"
)

plt.legend()
plt.savefig("./figs/BTC_backtest.png", dpi=1200)
plt.close()

# LUNA
series_luna['price_luna'].plot(label="actual LUNA")  # price_luna, volume_luna
backtest_series_luna_ust_inverse_trasform['price_luna'].plot(  # backtest_series.plot
    label=f"forecast LUNA",
    low_quantile=0.05,
    high_quantile=0.95  # Plot the median, 5th and 95th percentiles
)

plt.legend()
plt.savefig("./figs/LUNA_backtest.png", dpi=1200)
plt.close()

# UST
series_ust['price_ust'].plot(label="actual UST")  # price_ust, volume_ust
backtest_series_luna_ust_inverse_trasform['price_ust'].plot(  # backtest_series.plot
    label=f"forecast UST",
    low_quantile=0.05,
    high_quantile=0.95  # Plot the median, 5th and 95th percentiles
)

plt.legend()
plt.savefig("./figs/UST_backtest.png", dpi=1200)
plt.close()
