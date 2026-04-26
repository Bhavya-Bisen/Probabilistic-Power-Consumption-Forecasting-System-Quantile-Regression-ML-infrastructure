import mlflow
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import yaml
import os
from src.util.losses import MultiQuantileLoss
import mlflow.artifacts
import os
import mlflow.tensorflow
# -----------------------------
# Load config
# -----------------------------
config = yaml.safe_load(open("config/config.yaml"))

look_back = config["data"]["look_back"]
batch_size = config["training"]["batch_size"]
target_col = config["data"]["target_col"]
quantiles = config["loss"]["quantile_loss"]["multi_quantiles"]

# -----------------------------
# Load preprocessing artifacts
# -----------------------------
experiment = mlflow.get_experiment_by_name("Default")

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.stage = 'preprocessing'",
    order_by=["start_time DESC"],
    max_results=1
)

run_id = runs.iloc[0].run_id

train_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/train.pkl"
)

test_path = mlflow.artifacts.download_artifacts(
    run_id=run_id,
    artifact_path="data/test.pkl"
)

train_series = pd.read_pickle(train_path)
test_series = pd.read_pickle(test_path)

# -----------------------------
# Dataset builder
# -----------------------------
def make_window_dataset(series, look_back, batch_size, target_col=0):
    ds = tf.data.Dataset.from_tensor_slices(series)

    ds = ds.window(look_back + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(look_back + 1))

    ds = ds.map(lambda w: (w[:look_back, :], w[look_back, target_col]))

    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

test_ds = make_window_dataset(test_series, look_back, batch_size, target_col)

# -----------------------------
# Load model
# -----------------------------
local_path = mlflow.artifacts.download_artifacts(
    artifact_uri="runs:/c6dc7759a3994e66969082a48373560e/model"
)

model = tf.keras.models.load_model(
    os.path.join(local_path, "data/model"),
    custom_objects={"MultiQuantileLoss": MultiQuantileLoss}
)

# -----------------------------
# Prediction loop
# -----------------------------
# Predict efficiently using dataset
preds = model.predict(test_ds)

# Extract true values
y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

# -----------------------------
# Quantile split
# -----------------------------
q_to_index = {q: i for i, q in enumerate(quantiles)}

p10 = preds[:, q_to_index[0.1]]
p50 = preds[:, q_to_index[0.5]]
p90 = preds[:, q_to_index[0.9]]
# -----------------------------
# Inverse transform
# -----------------------------
scaler = MinMaxScaler()
scaler.fit(train_series)

def inverse_transform_target(y_scaled, scaler, target_col, n_features):
    # Create dummy array with full feature size
    dummy = np.zeros((len(y_scaled), n_features), dtype=np.float32)

    # Put target column back
    dummy[:, target_col] = y_scaled

    # Inverse scale
    inv = scaler.inverse_transform(dummy)

    # Extract target column
    return np.expm1(inv[:, target_col])

n_features = train_series.shape[1]

y_true_inv = inverse_transform_target(y_true, scaler, target_col, n_features)
p10_inv    = inverse_transform_target(p10, scaler, target_col, n_features)
p50_inv    = inverse_transform_target(p50, scaler, target_col, n_features)
p90_inv    = inverse_transform_target(p90, scaler, target_col, n_features)
# -----------------------------
# Metrics
# -----------------------------
mae = mean_absolute_error(y_true_inv, p50_inv)

coverage = np.mean(
    (y_true_inv >= p10_inv) & (y_true_inv <= p90_inv)
)

print(f"MAE (P50): {mae:.4f}")
print(f"Coverage (P10–P90): {coverage:.4%}")

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 6))

plt.plot(y_true_inv, label="Actual")
plt.plot(p50_inv, label="P50")

plt.fill_between(
    range(len(y_true_inv)),
    p10_inv,
    p90_inv,
    alpha=0.3,
    label="P10–P90"
)

plt.legend()
plt.title("Quantile Forecast Evaluation")

plot_path = "evaluation_plot.png"
plt.savefig(plot_path)
plt.close()

-----------------------------
Log to MLflow
# -----------------------------
with mlflow.start_run(run_name="evaluation"):

    mlflow.set_tag("stage", "evaluation")

    mlflow.log_metric("MAE_P50", mae)
    mlflow.log_metric("coverage_P10_P90", coverage)

    mlflow.log_artifact(plot_path)

    print("Logged evaluation to MLflow")