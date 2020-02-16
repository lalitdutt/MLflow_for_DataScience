__author__ = "lalitdutt parsai"
import pickle


class Client:
    """
    Scoring Client
    """
    def __init__(self,model_location):
        """
        Initialize Client with model location and Load model in model
        :param model_location:
        """
        self.model_location=model_location
        self.model=self.load_model()

    def load_model(self):
        """
        :return: Return Model
        """
        return pickle.load(open(self.model_location, 'rb'))

    def score(self,input_pd_dataframe):
        """
        Generate score using model
        :param input_pd_dataframe: Pandas data frame with features
        :return: Predicted Score
        """
        if self.model:
            return self.model.predict(input_pd_dataframe)

