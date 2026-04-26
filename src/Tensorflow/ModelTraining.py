from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow as tf

from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
import mlflow
import mlflow.tensorflow
import yaml
import pandas as pd
from src.util.losses import MultiQuantileLoss
# ---- Config ----
config = yaml.safe_load(open("/workspace/config.yaml"))

# get experiment
experiment = mlflow.get_experiment_by_name("Default")

# search runs
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'preprocessing'",
    order_by=["start_time DESC"],
    max_results=1
)
run_id = runs.iloc[0].run_id

# Get artifact paths
train_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="/workspace/data/train.pkl"
)

test_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="/workspace/data/test.pkl"
)

# Load data
train_series = pd.read_pickle(train_path)
test_series = pd.read_pickle(test_path)

def make_window_dataset(series,look_back, batch_size, target_col=0):
    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(
        look_back+1,
        shift=1,
        drop_remainder=True
    )

    ds = ds.flat_map(lambda w: w.batch(look_back+1))

    ds = ds.map(
        lambda w: (
            w[:look_back, :],          # X: all features, past L steps
            w[look_back, target_col]   # Y: target at time t
        )
    )

    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds

# reshape into X=t and Y=t+1
look_back = config["data"]["look_back"]
batch_size = config["training"]["batch_size"]
target_col= config["data"]["target_col"]
horizon= config["loss"]["quantile_loss"]["quantiles"]
train_ds = make_window_dataset(train_series, look_back, batch_size,target_col)
test_ds  = make_window_dataset(test_series, look_back, batch_size,target_col)

# ---- MLflow run ----
with mlflow.start_run(run_name=config["run_name"]):

    #Saving dataset as files and info
    mlflow.log_artifact("train.pkl", artifact_path="dataset")
    mlflow.log_artifact("test.pkl", artifact_path="dataset")
    mlflow.log_input(
        mlflow.data.from_pandas(train_series, source="train.pkl", name="train_dataset")
    )
    mlflow.log_input(
        mlflow.data.from_pandas(test_series, source="test.pkl", name="test_dataset")
    )

    # log config
    mlflow.log_params(config)

    mlflow.set_tag("stage", "training")

    # ---- MODEL ----
    model = Sequential()
    model.add(LSTM(config["model"]["lstm_units"], input_shape=(look_back, config["data"]["features"])))
    model.add(Dropout(config["model"]["dropout"]))
    model.add(Dense(config["loss"]["quantile_loss"]["quantiles"]))

    model.compile(
        loss=MultiQuantileLoss(
            config["loss"]["quantile_loss"]["multi_quantiles"]
            ),
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["training"]["learning_rate"],
            clipnorm=1.0
        )
    )

    # ---- CALLBACKS ----
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=config["early_stop"]["patience"],
        restore_best_weights=True
    )
    checkpoint=ModelCheckpoint(
        "checkpoint.keras",
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=False
    )

    # ---- TRAIN ----
    history = model.fit(
        train_ds,
        epochs=config["training"]["epochs"],
        validation_data=test_ds,
        callbacks=[early_stop,checkpoint],
        verbose=1,
        shuffle=False
    )

    # ---- LOG METRICS PER EPOCH ----
    for epoch in range(len(history.history["loss"])):
        mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)

    # ---- LOG FINAL METRICS ----
    mlflow.log_metric("final_train_loss", history.history["loss"][-1])
    mlflow.log_metric("final_val_loss", history.history["val_loss"][-1])

    # ---- LOG MODEL ----
    mlflow.tensorflow.log_model(model, "model")

    # ---- LOG MODEL SUMMARY ----
    with open("model_summary.txt", "w") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))

    mlflow.log_artifact("model_summary.txt")
