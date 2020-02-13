__author__ = "lalitdutt parsai"
import warnings
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

class WineQualityModel:
    """
    WineQuality Prediction Component ::: "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    """
    def __init__(self,dataset_location,alpha=0.5,l1_ratio=0.5,random_state=42):
        """
        Initialize Component
        :param dataset_location: Location of training dataset
        :param alpha: model pruning param alpha
        :param l1_ratio: model pruning param l1_ratio
        :param random_state: model pruning param random_state
        """
        self.train_dataset_location=dataset_location
        self.alpha=alpha
        self.l1_ratio=l1_ratio
        self.random_state=random_state

    def eval_metrics(self,actual, pred):
        """
        Metrics of Model
        :param actual: actual value
        :param pred: predicted value by model
        :return: rmse ,maem, r2
        """
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def read_dataset(self):
        """
        Read data
        :return: Pandas data frame of Dataset
        """
        try:
            return pd.read_csv(self.train_dataset_location, sep=';')
        except Exception as e:
            print("Unable to download training & test CSV, check your internet connection. Error: %s", e)

    def train_model(self):
        """
        Training of Model
        :return: Model with Metrics
        """
        warnings.filterwarnings("ignore")
        np.random.seed(40)
        data=self.read_dataset()
        #Split the data into training and test sets. (0.75, 0.25) split.
        train, test = train_test_split(data)

        #The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]
        lr = ElasticNet(self.alpha, self.l1_ratio, self.random_state)
        lr.fit(train_x, train_y)
        predicted_qualities = lr.predict(test_x)
        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (self.alpha, self.l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)
        return {"rmse":rmse,"mae":mae,"r2":r2,"model":lr}
