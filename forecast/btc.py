import logging
from datetime import datetime
from tabnanny import verbose
from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import BlockRNNModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape

from util import read_csv_with_ffill
from args import argparser

if __name__ == "__main__":
    """Hyper Params"""
    args = argparser()
    print("Params:", args)

    N_EPOCHS = args.epochs  # 300
    FORCE_RESET = args.reset  # False
    N_VAL = args.vals  # 21
    N_PRED = args.preds  # 360
    SEED = args.seed  # 950327
    LOAD = args.load  # True
    BEST = args.best  # True
    TRAIN = args.train  # False
    EVAL = args.eval  # False
    BACKTEST = args.test  # False
    PRED = args.pred  # False

    """Logging"""
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >> %(message)s')
    fileHandler = logging.FileHandler('./logs/btc.log')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    logger.info('Start: btc.py')
    logger.info(args)

    """Load Data"""
    csv_name = './data/btc-usd-max.csv'
    start_date = datetime(2020, 10, 2)
    df_btc = read_csv_with_ffill(csv_name, start_date)  # BTC-USD  # TODO: args

    logger.info("Read CSV: {}".format(csv_name))

    # Create a TimeSeries, specifying the time and value columns
    series_btc = TimeSeries.from_dataframe(df_btc, time_col='snapped_at', value_cols=['price', 'total_volume'], freq='D')
    series_btc = series_btc.with_columns_renamed(['price', 'total_volume'], ['price_btc', 'volume_btc'])

    # Set aside the last N_VAL days as a validation series
    # train, val = series.split_after(pd.Timestamp("20210101"))
    train_btc, val_btc = series_btc[:-N_VAL], series_btc[-N_VAL:]

    """Normalize"""
    # Normalize the time series (note: we avoid fitting the transformer on the validation set)
    transformer_btc = Scaler()

    train_btc_transformed = transformer_btc.fit_transform(train_btc)  # Fit at Training datasaet
    val_btc_transformed = transformer_btc.transform(val_btc)
    series_btc_transformed = transformer_btc.transform(series_btc)

    """DL"""
    if LOAD:
        model_btc = BlockRNNModel.load_from_checkpoint(model_name="BTC_GRU", best=BEST)
    else:
        # TODO: better RNN model
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
            # optimizer_cls # use Adam as default
            optimizer_kwargs={"lr": 1e-3},
            # loss_fn: MSE as default
            # likelihood=GaussianLikelihood(),  # ignore loss_fn
            # pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True},

            random_state=SEED,

            log_tensorboard=True,
            force_reset=FORCE_RESET,  # at darts_logs/
            save_checkpoints=True,
            show_warnings=False
        )

    if TRAIN:
        model_btc.fit(
            train_btc_transformed,
            val_series=val_btc_transformed,
            verbose=True
        )

    """Eval"""
    if EVAL:
        pred_series_btc = model_btc.predict(
            N_PRED,
            series=train_btc_transformed,
            num_samples=1,
            # verbose=False
        )

        mape_btc = mape(val_btc_transformed, pred_series_btc[:N_VAL])

        print("Evaluate: MAPE [BTC] = {:.3f}%".format(mape_btc))  # best at 0%
        logger.info("Evaluate: MAPE [BTC] = {:.3f}%".format(mape_btc))

        pred_series_btc_inverse_trasform = transformer_btc.inverse_transform(pred_series_btc)

        # Visualization
        series_btc['price_btc'].plot(label="actual BTC")
        pred_series_btc_inverse_trasform['price_btc'].plot(  # pred_series.plot
            label=f"forecast BTC"
        )
        plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)  # TODO: args
        plt.legend()

        fig_name = "./figs/BTC_evaluate.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

    """Backtest"""
    if BACKTEST:
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

        mape_btc = mape(series_btc_transformed, backtest_series_btc)

        print("Backtest: MAPE [BTC] = {:.3f}%".format(mape_btc))  # best at 0%
        logger.info("Backtest: MAPE [BTC] = {:.3f}%".format(mape_btc))

        backtest_series_btc_inverse_trasform = transformer_btc.inverse_transform(backtest_series_btc)
        backtest_series_btc_inverse_trasform = backtest_series_btc_inverse_trasform.with_columns_renamed(
            ['0', '1'],
            ['price_btc', 'volume_btc']
        )

        # Visualization
        series_btc['price_btc'].plot(label="actual BTC")  # price_btc, volume_btc
        backtest_series_btc_inverse_trasform['price_btc'].plot(  # backtest_series.plot
            label=f"forecast BTC"
        )
        plt.legend()

        fig_name = "./figs/BTC_backtest.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

    """Predict"""
    if PRED:
        # TODO: args
        custom_btc = np.array([  # Price, Volume
            [[482.3588], [9776900.705157386]],
            [[475.8200000000002], [19343751.807486504]],
            [[474.7263000000001], [14233663.71176932]],
            [[489.5475000000001], [23597497.056552686]],
            [[480.2425000000001], [14058803.04997312]],
            [[474.6938], [21083352.224831242]],
            [[480.6025], [7748654.382831849]],
        ])
        pred_series_custom_btc = TimeSeries.from_values(custom_btc)
        pred_series_custom_btc = transformer_btc.transform(pred_series_custom_btc)

        backtest_series_custom_btc = model_btc.predict(
            21,  # TODO: args
            series=pred_series_custom_btc,
            num_samples=1
        )

        backtest_series_custom_btc_inverse_trasform = transformer_btc.inverse_transform(backtest_series_custom_btc)
        backtest_series_custom_btc_inverse_trasform = backtest_series_custom_btc_inverse_trasform.with_columns_renamed(
            ['0', '1'],
            ['price_btc', 'volume_btc']
        )

        # Visualization
        backtest_series_custom_btc_inverse_trasform['price_btc'].plot(  # backtest_series.plot
            label=f"forecast BTC"
        )
        plt.legend()

        fig_name = "./figs/BTC_predict.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

    logger.info('End: btc.py')
