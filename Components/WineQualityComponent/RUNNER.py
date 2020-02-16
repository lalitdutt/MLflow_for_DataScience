__author__ = "lalitdutt parsai"
import os
import json
import pandas as pd
import mlflow.sklearn
from Components.WineQualityComponent.src.WineQualityModel.WineQualityTraining import WineQualityModel
from Components.WineQualityComponent.src.WineQualityClient.Client import Client
import logging
logging.basicConfig(level=logging.INFO)

cmd = "python src/setup.py sdist --dist-dir=artifact"


def runner(url,experiment_name,training_data_location,alpha,l1_ratio,test_data_location):
    if url is not None:
        mlflow.set_tracking_uri(url)
    if experiment_name is not None:
        mlflow.set_experiment(experiment_name)
    object = WineQualityModel(dataset_location=training_data_location,
                              alpha=alpha,
                              l1_ratio=l1_ratio)
    with mlflow.start_run():
        active_run=mlflow.active_run()
        if tracking_url is None:
            logging.info("Tracking_Server: running on Localhost")
        else:
            logging.info("Tracking_Server: running on Server")
        run_id = active_run.info.run_id
        logging.info("Experiment: " + experiment_name)
        logging.info("Model training -> -> ")
        result=object.train_model()
        mlflow.log_param("training_set", training_data_location)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", result["rmse"])
        mlflow.log_metric("r2", result["r2"])
        mlflow.log_metric("mae", result["mae"])
        mlflow.sklearn.log_model(result["model"], "model")
        model_location="mlruns/0/"+run_id+"/artifacts/model/model.pkl"
        test_data=pd.read_csv(test_data_location, sep=',')
        client=Client(model_location)
        data=client.score(test_data)
        logging.info("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
        logging.info("  RMSE: %s" % result["rmse"])
        logging.info("  MAE: %s" % result["mae"])
        logging.info("  R2: %s" % result["r2"])
        logging.info("End of model training ! ")
        if len(data)>1:
            mlflow.log_param("scoring_package", "passed")
            logging.info("Building scoring client!")
            os.system(cmd)  # returns the exit code in unix
            mlflow.log_artifact("artifact/WineQualityClient-0.0.1.tar.gz", "client")
        else:
            mlflow.log_param("scoring_package", "fail")
        return run_id


if __name__ == "__main__":
    #tracking_url = 'http://<your_http_server>:5000'
    tracking_url = None
    app_config = None

    with open('conf.json') as json_file:
        app_config = json.load(json_file)

    experiment_name = app_config['experiment_name']
    training_data_location = app_config['default_training_data']
    test_data_location=app_config['default_test_data']
    alpha = float(app_config['default_alpha'])
    l1_ratio = float(app_config['default_l1_ratio'])


    # logging.info("Experiment: " + experiment_name)
    run_id=runner(tracking_url,experiment_name,training_data_location, alpha, l1_ratio, test_data_location)
    logging.info("Run_id: "+ run_id)

