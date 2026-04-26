from mlflow.tracking import MlflowClient
import mlflow
import pandas as pd

run_id = "c6dc7759a3994e66969082a48373560e"

train_df = pd.read_pickle("train.pkl")
test_df = pd.read_pickle("test.pkl")


with mlflow.start_run(run_id=run_id):

    # # run-level metadata (recommended)
    # mlflow.set_tag("stage", "training")

    # # artifacts (actual files)
    # mlflow.log_artifact("train.pkl", artifact_path="dataset")
    # mlflow.log_artifact("test.pkl", artifact_path="dataset")

    # train_df=pd.DataFrame(train_df)
    # test_df=pd.DataFrame(test_df)
    # # structured dataset tracking
    # mlflow.log_input(
    #     mlflow.data.from_pandas(train_df, source="train.pkl", name="train_dataset")
    # )
    # mlflow.log_input(
    #     mlflow.data.from_pandas(test_df, source="test.pkl", name="test_dataset")
    # )

    mlflow.log_artifact("src/Tensorflow/DataPreprocessing.py")
    mlflow.log_artifact("src/Tensorflow/ModelEvaluation.py")
    mlflow.log_artifact("src/Tensorflow/ModelTraining.py")
    mlflow.log_artifact("src/util/losses.py")
    mlflow.log_artifact("config/config.yaml")