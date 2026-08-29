import mlflow
import mlflow.sklearn


def track_experiment(
        model,
        params: dict,
        metrics: dict,
        experiment_name:str = "default",
        run_name:str = "run_1"
):
    '''
    track the experiment of mlflow experiment
    '''

    # set the experiments and run 
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name) as run:

        # log the params and metrics
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)

        # log the model 
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        print(f"track the experiment:{experiment_name}, run name: {run_name}")

        return run.info.run_id