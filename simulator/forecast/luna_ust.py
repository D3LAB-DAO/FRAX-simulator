from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import BlockRNNModel
from darts.timeseries import concatenate
from darts.utils.likelihood_models import GaussianLikelihood

from util import read_csv_with_ffill


if __name__ == "__main__":
    from datetime import datetime
    import logging
    # from tabnanny import verbose
    # from tracemalloc import start
    import matplotlib.pyplot as plt
    import numpy as np

    from darts.metrics import mape

    from args import argparser

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
    fileHandler = logging.FileHandler('./logs/luna_ust.log')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    logger.info('Start: luna_ust.py')
    logger.info(args)

    """Load Data"""
    csv_name_btc = './data/btc-usd-max.csv'
    csv_name_luna = './data/luna-usd-max.csv'
    csv_name_ust = './data/ust-usd-max.csv'
    start_date = datetime(2020, 10, 2)

    df_btc = read_csv_with_ffill(csv_name_btc, start_date)  # TODO: args
    df_luna = read_csv_with_ffill(csv_name_luna, start_date)  # TODO: args
    df_ust = read_csv_with_ffill(csv_name_ust, start_date)  # TODO: args

    logger.info("Read CSV: {}".format(csv_name_btc))
    logger.info("Read CSV: {}".format(csv_name_luna))
    logger.info("Read CSV: {}".format(csv_name_ust))

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
    if LOAD:
        model_luna_ust = BlockRNNModel.load_from_checkpoint(model_name="LUNA_UST_GRU", best=BEST)
    else:
        # TODO: better RNN model
        model_luna_ust = BlockRNNModel(
            model="GRU",
            model_name="LUNA_UST_GRU",

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
            likelihood=GaussianLikelihood(),  # ignore loss_fn
            # pl_trainer_kwargs={"accelerator": "gpu", "gpus": -1, "auto_select_gpus": True},

            random_state=SEED,

            log_tensorboard=True,
            force_reset=FORCE_RESET,  # at darts_logs/
            save_checkpoints=True,
            show_warnings=False
        )

    if TRAIN:
        model_luna_ust.fit(
            train_luna_ust_transformed,
            past_covariates=train_btc_transformed,
            val_series=val_luna_ust_transformed,
            val_past_covariates=val_btc_transformed,
            verbose=True
        )

    """Eval"""
    if EVAL:
        # Load BTC
        model_btc = BlockRNNModel.load_from_checkpoint(model_name="BTC_GRU", best=BEST)
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

        mape_luna_ust = mape(val_luna_ust_transformed, pred_series_luna_ust[:N_VAL])

        print("Evaluate: MAPE [LUNA_UST] = {:.3f}%".format(mape_luna_ust))  # best at 0%
        logger.info("Evaluate: MAPE [LUNA_UST] = {:.3f}%".format(mape_luna_ust))

        pred_series_luna_ust_inverse_trasform = transformer_luna_ust.inverse_transform(pred_series_luna_ust)

        # Visualization LUNA
        series_luna['price_luna'].plot(label="actual LUNA")
        pred_series_luna_ust_inverse_trasform['price_luna'].plot(  # pred_series.plot
            label=f"forecast LUNA",
            low_quantile=0.05,
            high_quantile=0.95  # Plot the median, 5th and 95th percentiles
        )
        plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)  # TODO: args
        plt.legend()

        fig_name = "./figs/LUNA_evaluate.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

        # Visualization UST
        series_ust['price_ust'].plot(label="actual UST")
        pred_series_luna_ust_inverse_trasform['price_ust'].plot(  # pred_series.plot
            label=f"forecast UST",
            low_quantile=0.05,
            high_quantile=0.95  # Plot the median, 5th and 95th percentiles
        )
        plt.axvline(x=datetime(2022, 4, 16), color='r', linestyle='--', linewidth=0.5)  # TODO: args
        plt.legend()

        fig_name = "./figs/UST_evaluate.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

    """Backtest"""
    if BACKTEST:
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

        mape_luna_ust = mape(series_luna_ust_transformed, backtest_series_luna_ust)

        print("Backtest: MAPE [LUNA_UST] = {:.3f}%".format(mape_luna_ust))  # best at 0%
        logger.info("Backtest: MAPE [LUNA_UST] = {:.3f}%".format(mape_luna_ust))

        backtest_series_luna_ust_inverse_trasform = transformer_luna_ust.inverse_transform(backtest_series_luna_ust)
        backtest_series_luna_ust_inverse_trasform = backtest_series_luna_ust_inverse_trasform.with_columns_renamed(
            ['0', '1', '2', '3'],
            ['price_luna', 'volume_luna', 'price_ust', 'volume_ust']
        )

        # Visualization LUNA
        series_luna['price_luna'].plot(label="actual LUNA")  # price_luna, volume_luna
        backtest_series_luna_ust_inverse_trasform['price_luna'].plot(  # backtest_series.plot
            label=f"forecast LUNA",
            low_quantile=0.05,
            high_quantile=0.95  # Plot the median, 5th and 95th percentiles
        )
        plt.legend()

        fig_name = "./figs/LUNA_backtest.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

        # Visualization UST
        series_ust['price_ust'].plot(label="actual UST")  # price_ust, volume_ust
        backtest_series_luna_ust_inverse_trasform['price_ust'].plot(  # backtest_series.plot
            label=f"forecast UST",
            low_quantile=0.05,
            high_quantile=0.95  # Plot the median, 5th and 95th percentiles
        )
        plt.legend()

        fig_name = "./figs/UST_backtest.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

    """Predict"""
    if PRED:
        # TODO: args
        custom_btc_raw = np.array([  # Price, Volume
            [[482.3588], [9776900.705157386]],
            [[475.8200000000002], [19343751.807486504]],
            [[474.7263000000001], [14233663.71176932]],
            [[489.5475000000001], [23597497.056552686]],
            [[480.2425000000001], [14058803.04997312]],
            [[474.6938], [21083352.224831242]],
            [[480.6025], [7748654.382831849]],
        ])
        series_custom_btc_raw = TimeSeries.from_values(custom_btc_raw)
        series_custom_btc = transformer_btc.transform(series_custom_btc_raw)
        
        custom_luna_ust_raw = np.array([  # ['price_luna', 'volume_luna', 'price_ust', 'volume_ust']
            [[97.23776575054052], [2021140001.2010279], [1.0009731813465388], [658286545.56903]],
            [[92.58947577407795], [1532006458.5447805], [1.0012466506069766], [534537541.69056]],
            [[82.39863959874687], [3180497426.094548], [1.0012959754315212], [1154542537.8449473]],
            [[84.57206595034742], [2406029472.5926356], [1.0043964951114448], [825356747.4021069]],
            [[87.84777221968402], [1870303989.9037478], [1.001990283818642], [626240897.9884915]],
            [[81.67575260339214], [1877453895.6330187], [1.0024502187641433], [629390555.732739]],
            [[80.3496141479117], [1215927530.411061], [1.0033201986661755], [458162984.4615695]],
        ])
        series_custom_luna_ust_raw = TimeSeries.from_values(custom_luna_ust_raw)
        series_custom_luna_ust = transformer_luna_ust.transform(series_custom_luna_ust_raw)

        # Load BTC
        model_btc = BlockRNNModel.load_from_checkpoint(model_name="BTC_GRU", best=BEST)
        pred_series_btc = model_btc.predict(
            21,  # 20
            series=series_custom_btc,
            num_samples=1,
            # verbose=False
        )

        pred_series_custom_luna_ust = model_luna_ust.predict(
            21,  # TODO: args
            series=series_custom_luna_ust,
            past_covariates=concatenate([series_custom_btc, pred_series_btc[-21:]], axis=0),
            num_samples=1000,
            # verbose=False
        )

        pred_series_custom_luna_ust_inverse_trasform = transformer_luna_ust.inverse_transform(pred_series_custom_luna_ust)
        pred_series_custom_luna_ust_inverse_trasform = pred_series_custom_luna_ust_inverse_trasform.with_columns_renamed(
            ['0', '1', '2', '3'],
            ['price_luna', 'volume_luna', 'price_ust', 'volume_ust']
        )

        # Visualization LUNA
        pred_series_custom_luna_ust_inverse_trasform['price_luna'].plot(
            label=f"forecast LUNA",
            low_quantile=0.05,
            high_quantile=0.95  # Plot the median, 5th and 95th percentiles
        )
        plt.legend()

        fig_name = "./figs/LUNA_predict.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

        # Visualization UST
        pred_series_custom_luna_ust_inverse_trasform['price_ust'].plot(
            label=f"forecast UST",
            low_quantile=0.05,
            high_quantile=0.95  # Plot the median, 5th and 95th percentiles
        )
        plt.legend()

        fig_name = "./figs/UST_predict.png"

        plt.savefig(fig_name, dpi=1200)
        plt.close()

        logger.info("Save Fig: {}".format(fig_name))

    logger.info('End: luna_ust.py')
