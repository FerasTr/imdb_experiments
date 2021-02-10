import numpy as np
from readData import imdb_data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MultiLabelBinarizer


def prepare_inputs(data):
    mlb = MultiLabelBinarizer()
    x_data1 = pd.DataFrame(
        mlb.fit_transform(data.pop("director").str.split(",")),
        columns=mlb.classes_,
        index=data.index,
    ).add_prefix("dir_")
    x_data2 = pd.DataFrame(
        mlb.fit_transform(data.pop("actors").str.split(",")),
        columns=mlb.classes_,
        index=data.index,
    ).add_prefix("act_")
    return x_data1.join(x_data2)


if __name__ == "__main__":
    data = imdb_data()
    x_data = prepare_inputs(data)
    y_data = data.pop("avg_vote")
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    print(x_train)
    print(y_train)
    print(x_test)
    print(y_test)
    mlpr = MLPRegressor(random_state=1, max_iter=100)
    mlpr.fit(x_train, y_train)
    prediction = mlpr.predict(x_test)
    print(prediction)
    print(y_test)
    score = mlpr.score(x_test, y_test)
    print(score)
