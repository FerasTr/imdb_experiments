import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split


class imdb_data:
    def __init__(self, file="imdb_movies.csv"):
        self.data = self._imdb_data(file)

    def get_encoded_train_test(self):
        """
        Split the data into 80/20 training/test sets

        Returns
        ---------
        x_train: training set
        x_test: testing set
        y_train: training set's labels
        y_test: testing set's labels
        """
        x_data = self._get_encoded_data()
        y_data = self.data.pop("avg_vote")
        print("Splitting data...")
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=0.2
        )
        return x_train, x_test, y_train, y_test

    def _get_encoded_data(self):
        """
        Encodes the data using one hot encoding, where each director and actor
        becomes a column

        Returns
        ---------
        A one-hot-encoding DataFrame object of 'director' and 'actors' columns
        """
        print("Encoding data...")
        mlb = MultiLabelBinarizer()
        x_data1 = pd.DataFrame(
            mlb.fit_transform(self.data.pop("director").str.split(",")),
            columns=mlb.classes_,
            index=self.data.index,
        ).add_prefix("dir_")
        x_data2 = pd.DataFrame(
            mlb.fit_transform(self.data.pop("actors").str.split(",")),
            columns=mlb.classes_,
            index=self.data.index,
        ).add_prefix("act_")
        return x_data1.join(x_data2)

    def _imdb_data(self, file):
        """
        Preprocessing for the imdb dataset

        Parameters
        ---------
        file: csv file for the imdb movies dataset

        Returns
        ---------
        A DataFrame object of 'director', 'actors' and 'avg_rating' columns
        """

        print("Preparing data...")
        # Read data
        imdb = pd.DataFrame(
            data=pd.read_csv(
                file, engine="python", delimiter=",", index_col="imdb_title_id"
            )
        )
        # Remove rows where one element is missing
        imdb.dropna(inplace=True)
        # Remove non English movies
        imdb = imdb[imdb.language == "English"]
        # Convert year col to numeric
        imdb = imdb[imdb.year.apply(lambda x: x.isnumeric())]
        imdb["year"] = pd.to_numeric(imdb["year"])
        # Remove movies older than 1980
        imdb = imdb[imdb.year >= 1980]
        # Remove unused columns
        imdb.drop(
            columns=[
                "title",
                "original_title",
                "date_published",
                "genre",
                "duration",
                "country",
                "writer",
                "production_company",
                "description",
                "budget",
                "usa_gross_income",
                "worlwide_gross_income",
                "metascore",
                "reviews_from_users",
                "reviews_from_critics",
                "votes",
                "year",
                "language",
            ],
            inplace=True,
        )
        # Limit director/actor list of each movie to one director/4 actors
        imdb.actors = imdb.actors.apply(
            lambda r: (",".join([x.strip() for x in r.split(",")[:3]])).strip()
        )
        imdb.director = imdb.director.apply(
            lambda r: (",".join([x.strip() for x in r.split(",")[:1]])).strip()
        )
        # Remove duplicate lines
        imdb.drop_duplicates(inplace=True)
        return imdb


if __name__ == "__main__":
    # Testing data structure
    data = imdb_data()
    print(data.data.loc["tt9173418"])
    print(data.data.head())
    print(data.data.tail())
    print(data.data.info())
    print(data.data.shape)
    x_train, x_test, y_train, y_test = data.get_encoded_train_test()
    print(x_train)
    print(y_train)
