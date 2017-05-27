from utils import utils
from data_preprocessing import normalization

from sklearn import datasets

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def mlp_classifier():
    """
    This function show the use of a neural network with iris dataset
    """

    credit = datasets.load_credit()
    data_features = credit.data
    data_targets = credit.target

    # Data normalization
    data_features_normalized = normalization.z_score_normalization(data_features)

    # Data splitting
    data_features_train, data_features_test, data_targets_train, data_targets_test = utils.data_splitting(
        data_features_normalized,
        data_targets,
        0.25)
    neural_net = MLPClassifier(
        hidden_layer_sizes=(50),
        activation="relu",
        solver="adam"
    )
    neural_net.fit(data_features_train, data_targets_train)

    # Model evaluation
    test_data_predicted = neural_net.predict(data_features_test)
    score = metrics.accuracy_score(data_targets_test, test_data_predicted)

    logger.debug("Model Score: %s", score)


if __name__ == '__main__':
    logger.info("###################---MLP Classifier---###################")
    # Classification example
    mlp_classifier()

    #logger.info("###################---MLP Regressor---###################")
    # Regression example
