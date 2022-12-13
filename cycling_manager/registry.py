import glob
import os
import time
import pickle

import mlflow
from mlflow.tracking import MlflowClient

from colorama import Fore, Style

from tensorflow.keras import Model, models

def save_model(model: Model = None,
               params: dict = None,
               metrics: dict = None) -> None:
    """
    persist trained model, params and metrics
    """

    timestamp = time.strftime("%Y%m%d-%H%M%S")

    if os.environ.get("MODEL_TARGET") == "mlflow":

        # retrieve mlflow env params
        # $CHA_BEGIN
        mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
        # $CHA_END

        # configure mlflow
        # $CHA_BEGIN
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(experiment_name=mlflow_experiment)
        # $CHA_END

        with mlflow.start_run():

            # STEP 1: push parameters to mlflow
            # $CHA_BEGIN
            if params is not None:
                mlflow.log_params(params)
            # $CHA_END

            # STEP 2: push metrics to mlflow
            # $CHA_BEGIN
            if metrics is not None:
                mlflow.log_metrics(metrics)
            # $CHA_END

            # STEP 3: push model to mlflow
            # $CHA_BEGIN
            if model is not None:

                mlflow.keras.log_model(keras_model=model,
                                       artifact_path="model",
                                       keras_module="tensorflow.keras",
                                       registered_model_name=mlflow_model_name)
            # $CHA_END

        # $WIPE_BEGIN
        print("\n✅ data saved to mlflow")
        # $WIPE_END

        return None

    print(Fore.BLUE + "\nSave model to local disk..." + Style.RESET_ALL)

    # save params
    if params is not None:
        params_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "params", timestamp + ".pickle")
        print(f"- params path: {params_path}")
        with open(params_path, "wb") as file:
            pickle.dump(params, file)

    # save metrics
    if metrics is not None:
        metrics_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "metrics", timestamp + ".pickle")
        print(f"- metrics path: {metrics_path}")
        with open(metrics_path, "wb") as file:
            pickle.dump(metrics, file)

    # save model
    if model is not None:
        model_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models", timestamp)
        print(f"- model path: {model_path}")
        model.save(model_path)

    print("\n✅ data saved locally")

    return None