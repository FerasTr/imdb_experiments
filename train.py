import numpy as np
from readData import imdb_data
import pandas as pd
from sklearn.pipeline import make_pipeline

from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler


class Train:
    def __init__(self):
        self.data = imdb_data()
        (
            self.x_train,
            self.x_test,
            self.y_train,
            self.y_test,
        ) = self.data.get_encoded_train_test()
        self.final_results = self.y_test.copy(deep=True).to_frame()
        self.scores = []

    def _saveResults(self):
        """
        Saves the final results of the model in two files.
        1) A csv with all of the results
        2) and a smaller summary text file.
        """
        print("Saving results...")
        self.final_results.to_csv("final_results.csv")
        with open("summary.txt", "w+") as file_summary:
            print(self.final_results.head(), file=file_summary)
            print(self.final_results.tail(), file=file_summary)
            print(self.scores, file=file_summary)

    def train(self):
        """
        Main training method, trains multiple methods with different parameters
        """
        print("Training...")

        iterations = [200, 400, 600]
        neighs = [3, 6, 10, 20, 40]

        for _neigh in neighs:
            self._KNN(neighbors=_neigh)

        for _iter in iterations:
            self._mlp(iterations=_iter)
            self._svr(iterations=_iter)
            self._bays_ridge(iterations=_iter)

        # Save model results
        self._saveResults()
        print("Done...")

    def _mlp(self, iterations=100):
        """
        Multi-layer Perceptron learning algorithm

        Parameters
        ---------
        iterations: number of iterations to run the algortithm for

        Returns
        ---------
        results: Series containing the results indexed by imdb id

        dictonary containing the name of the experiment and the accuracy
        """
        print("mlp model")
        name = f"mlp_{iterations}"
        mlpr = MLPRegressor(
            random_state=1,
            max_iter=iterations,
            activation="tanh",
            learning_rate="adaptive",
            solver="sgd",
            learning_rate_init=0.001,
            alpha=0.01,
            early_stopping=True,
            verbose=True,
        )
        self._model_train(name, mlpr)

    def _KNN(self, neighbors=10):
        """
        K-Nearest Neighbors learning algorithm

        Parameters
        ---------
        neighbors: number of neighbors to use in learning
        """
        print("KNN model")
        name = f"knn_{neighbors}"
        neigh = KNeighborsRegressor(
            n_neighbors=neighbors,
            algorithm="kd_tree",
            weights="distance",
            p=2,
            metric="minkowski",
        )
        self._model_train(name, neigh)

    def _bays_ridge(self, iterations=100):
        """
        Bayesian ridge learning algorithm

        Parameters
        ---------
        iterations: number of iterations to run the algortithm for
        """
        print("Bayes ridge model")
        name = f"baysRidge_{iterations}"
        rig = linear_model.BayesianRidge(n_iter=iterations, verbose=True)
        self._model_train(name, rig)

    def _svr(self, iterations=100):
        """
        Support Vector Regression learning algorithm

        Parameters
        ---------
        iterations: number of iterations to run the algortithm for
        """
        print("SVR model")
        name = f"svr_{iterations}"
        svr = make_pipeline(
            StandardScaler(),
            SVR(
                kernel="linear", C=1.0, epsilon=0.001, max_iter=iterations, verbose=True
            ),
        )
        self._model_train(name, svr)

    def _model_train(self, name, model):
        """
        Fit a model to the train data and compute the score on the test data

        Parameters
        ---------
        name: model's name
        model: model to train
        """
        model.fit(self.x_train, self.y_train)
        prediction = model.predict(self.x_test)
        score = model.score(self.x_test, self.y_test)
        results = pd.Series(prediction, name=name, index=self.final_results.index)
        self.scores.append({name: score})
        self.final_results = self.final_results.join(results)


if __name__ == "__main__":
    trainer = Train()
    trainer.train()
