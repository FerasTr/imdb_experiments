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
            print(file_summary.read())

    def train(self):
        """
        Main training method, trains multiple methods with different parameters
        """
        print("Training...")
        self.scores = []

        iterations = [100, 300, 500]
        neighs = [3, 6, 10, 20, 40]

        for _neigh in neighs:
            res, sc = self._KNN(neighbors=_neigh)
            self.scores.append(sc)
            self.final_results = self.final_results.join(res)

        for _iter in iterations:
            res, sc = self._mlp(iterations=_iter)
            self.scores.append(sc)
            self.final_results = self.final_results.join(res)

            res, sc = self._svr(iterations=_iter)
            self.scores.append(sc)
            self.final_results = self.final_results.join(res)

            res, sc = self._bays_ridge(iterations=_iter)
            self.scores.append(sc)
            self.final_results = self.final_results.join(res)

        # Save model results
        self.saveResults()
        print("Done...")

    # TODO imporve each model
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
        name = f"mlp_{iterations}"
        mlpr = MLPRegressor(random_state=1, max_iter=iterations)
        mlpr.fit(self.x_train, self.y_train)
        prediction = mlpr.predict(self.x_test)
        results = pd.Series(prediction, name=name, index=self.final_results.index)
        score = mlpr.score(self.x_test, self.y_test)
        return results, {name: score}

    def _KNN(self, neighbors=10):
        """
        K-Nearest Neighbors learning algorithm

        Parameters
        ---------
        neighbors: number of neighbors to use in learning

        Returns
        ---------
        results: Series containing the results indexed by imdb id

        dictonary containing the name of the experiment and the accuracy
        """
        name = f"knn_{neighbors}"
        neigh = KNeighborsRegressor(n_neighbors=neighbors, algorithm="auto")
        neigh.fit(self.x_train, self.y_train)
        prediction = neigh.predict(self.x_test)
        score = neigh.score(self.x_test, self.y_test)
        results = pd.Series(prediction, name=name, index=self.final_results.index)
        return results, {name: score}

    def _bays_ridge(self, iterations=100):
        """
        Bayesian ridge learning algorithm

        Parameters
        ---------
        iterations: number of iterations to run the algortithm for

        Returns
        ---------
        results: Series containing the results indexed by imdb id

        dictonary containing the name of the experiment and the accuracy
        """
        name = f"baysRidge_{iterations}"
        rig = linear_model.BayesianRidge(n_iter=iterations)
        rig.fit(self.x_train, self.y_train)
        prediction = rig.predict(self.x_test)
        score = rig.score(self.x_test, self.y_test)
        results = pd.Series(prediction, name=name, index=self.final_results.index)
        return results, {name: score}

    def _svr(self, iterations=100):
        """
        Support Vector Regression learning algorithm

        Parameters
        ---------
        iterations: number of iterations to run the algortithm for

        Returns
        ---------
        results: Series containing the results indexed by imdb id

        dictonary containing the name of the experiment and the accuracy
        """
        name = f"svr_{iterations}"
        svr = SVR(max_iter=iterations)
        svr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        svr.fit(self.x_train, self.y_train)
        prediction = svr.predict(self.x_test)
        score = svr.score(self.x_test, self.y_test)
        results = pd.Series(prediction, name=name, index=self.final_results.index)
        return results, {name: score}


if __name__ == "__main__":
    trainer = Train()
    trainer.train()
