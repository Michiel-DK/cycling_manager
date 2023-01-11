import glob
import os
import time
import pickle

        #mlflow_t
import logging
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import boto3
import mlflow
from mlflow.data import parse_s3_uri
from mlflow.entities import Run

import mlflow
from mlflow.tracking import MlflowClient

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

from colorama import Fore, Style

from tensorflow.keras import Model, models

logger = logging.getLogger(__name__)

from botocore.config import Config

my_config = Config(
    region_name = 'us-west-1',
)

def ensure_s3_bucket_for_run(run: Run):
    """Create bucket if not exists.
    :param run: MLFlow run
    """
    try:
        bucket, __ = parse_s3_uri(run.info.artifact_uri)
    except Exception:
        pass
    else:
        s3_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        print(s3_endpoint_url)
        if s3_endpoint_url:
            s3_client = boto3.client("s3", endpoint_url=s3_endpoint_url, config=my_config)
            try:
                s3_client.create_bucket(Bucket=bucket)
            except (
                s3_client.exceptions.BucketAlreadyExists,
                s3_client.exceptions.BucketAlreadyOwnedByYou,
            ):
                logger.info("Bucket already exists", bucket=bucket)
                print("Bucket already exists")


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
        #mlflow_tracking_uri = 'postgresql+psycopg2://postgres:postgres@localhost/mlflow_db'
        if not mlflow.get_experiment_by_name(os.environ.get("MLFLOW_EXPERIMENT")):
            mlflow.create_experiment(os.environ.get("MLFLOW_EXPERIMENT"), artifact_location=os.environ.get("ARTIFACT_LOCATION"))
        mlflow.set_tracking_uri( os.environ.get("MLFLOW_TRACKING_URI"))

        #mlflow_experiment = os.environ.get("MLFLOW_EXPERIMENT")
        # try:
        #     mlflow.create_experiment(mlflow_experiment, artifact_location="s3://mlflow")
        # except MlflowException as e:
        #     print(e)
        mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT"))
        #mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
        
        
        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
        # $CHA_END

        with mlflow.start_run() as run:

            # STEP 1: push parameters to mlflow
            # $CHA_BEGIN
            #ensure_s3_bucket_for_run(run)
            
            
            
            
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

def load_model(save_copy_locally=False) -> Model:
    """
    load the latest saved model, return None if no model found
    """
    if os.environ.get("MODEL_TARGET") == "mlflow":

        print(Fore.BLUE + f"\nLoad model stage from mlflow..." + Style.RESET_ALL)

        # load model from mlflow
        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")
        
                #model_uri = f"models:/{mlflow_model_name}/{stage}"
        logged_model = 'runs:/26297aae497143b7a75464a52ad3a4da/model'
        
        #model_uri = f"models:/{mlflow_model_name}/{stage}"
        model_uri = f'{os.environ.get("ARTIFACT_LOCATION")}//artifacts/model'
        print(f"- uri: {model_uri}")

        try:
            model = mlflow.keras.load_model(logged_model)
            print("\n✅ model loaded from mlflow")
        except:
            print(f"\n❌ no model in stage on mlflow")
            return None

        if save_copy_locally:
            from pathlib import Path

            # Create the LOCAL_REGISTRY_PATH directory if it does exist
            Path(os.environ.get("LOCAL_REGISTRY_PATH")).mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            model_path = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models", timestamp)
            model.save(model_path)

        return model

    print(Fore.BLUE + "\nLoad model from local disk..." + Style.RESET_ALL)

    # get latest model version
    model_directory = os.path.join(os.environ.get("LOCAL_REGISTRY_PATH"), "models")

    results = glob.glob(f"{model_directory}/*")
    if not results:
        return None

    model_path = sorted(results)[-1]
    print(f"- path: {model_path}")

    model = models.load_model(model_path)
    print("\n✅ model loaded from disk")

    return model

def get_model_version(stage="Production"):
    """
    Retrieve the version number of the latest model in the given stage
    - stages: "None", "Production", "Staging", "Archived"
    """

    if os.environ.get("MODEL_TARGET") == "mlflow":

        mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))

        mlflow_model_name = os.environ.get("MLFLOW_MODEL_NAME")

        client = MlflowClient()

        try:
            version = client.get_latest_versions(name=mlflow_model_name, stages=[stage])
        except:
            return None

        # check whether a version of the model exists in the given stage
        if not version:
            return None

        return int(version[0].version)

    # model version not handled

    return None
