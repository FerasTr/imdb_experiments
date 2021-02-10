import pandas as pd


def imdb_data(file="imdb_movies.csv"):
    """
    Preprocessing for the imdb dataset

    Parameters
    ---------
    file: csv file for the imdb movies dataset

    Returns
    ---------
    Returns a pandas object containing columns for: movies, actors, directors, ratings
    """

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
    data = imdb_data()
    print(data.loc['tt9173418'])
    print(data.head())
    print(data.tail())
    print(data.info())
    print(data.shape)
