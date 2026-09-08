import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient


def model_register(
        model,
        model_name: str,
        experiment_name: str = "default",
        run_name: str = "run_1",
        alias: str = "production"
):
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name
        )

        client = MlflowClient()

        latest_version = model_info.registered_model_version

        client.set_registered_model_alias(
            name=model_name,
            version=latest_version,
            alias=alias
        )

        print(
            f"model registered: {model_name}, "
            f"latest version: {latest_version}, "
            f"alias: {alias}"
        )

        return latest_version