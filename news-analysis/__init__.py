import logging
import os
import json

import pymongo
from urllib.request import urlopen, Request
from bs4 import BeautifulSoup
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from datetime import datetime, timedelta, timezone
import tensorflow as tf

# Azure imports
import azure.functions as func
from azure.storage.blob import BlobServiceClient

# Definitions
nltk.download("vader_lexicon")
finviz_url = "https://finviz.com/quote.ashx?t="


def download_model_and_word_index():
    """
    Downloads the prediction model and the word index from an predefined azure blob storage.
    Updates to this blob storage can be made via the stockalizer project
    that can be found here: https://github.com/SoenkeSobott/stockalizer

    Returns:
        The prediction model and the word index
    """
    # Get client
    blob_service_client = BlobServiceClient.from_connection_string(
        os.environ["BLOB_CONNECTION_STRING"]
    )
    container_client = blob_service_client.get_container_client("models")

    # Instantiate a new BlobClient
    blob_client = container_client.get_blob_client("SentimentAnalysis")
    blob_client_word_index = container_client.get_blob_client("WordIndex")

    # Download model
    with open("/tmp/text-classification-blob.h5", "wb") as my_blob:
        download_stream = blob_client.download_blob()
        my_blob.write(download_stream.readall())

    # Download word index
    with open("/tmp/word-index.json", "wb") as my_blob:
        download_stream = blob_client_word_index.download_blob()
        my_blob.write(download_stream.readall())

    with open("/tmp/word-index.json") as json_file:
        word_index = json.load(json_file)

    # Return model and word_index
    return tf.keras.models.load_model("/tmp/text-classification-blob.h5"), word_index


# Get model and word_index
model, word_index = download_model_and_word_index()


def encode_title(title):
    """
    Encodes the passed array. This means that the words from the array are mapped
    to their referenced values in the word_index dictionary.
    Example: in: ["this", "is", "a", "test"]   out: [3244, 345, 4, 23444]

    Args:
        title: The array with all the words that need to be encoded.

    Returns:
        The encoded array.
    """
    encoded = [1]
    for word in title:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


def prepare_news_title(title):
    """
    Prepares the passed text. That means removing unnecessary chars and then transforming it to an array and
    encoding the array. Also the maximum length is set and all empty spaces to this maximum are filled with the
    placeholder value <PAD>.

    Args:
        title: The text that needs preparation.

    Returns:
        The encoded array.
    """
    title = (
        title.replace(".", "").replace(",", "").replace(":", "").split()
    )  # remove more values !?*"" ...
    encoded_title = encode_title(title)
    encoded_title = tf.keras.preprocessing.sequence.pad_sequences(
        [encoded_title], value=word_index["<PAD>"], padding="post", maxlen=100
    )
    return encoded_title


def get_news_table_for_ticker(ticker):
    """
    Extracts the news table for the specified ticker from the finviz website

    Args:
        ticker: The ticker from which the news table should be extracted.

    Returns:
        The news table.
    """
    url = finviz_url + ticker

    req = Request(url=url, headers={"user-agent": "my-app"})
    response = urlopen(req)
    html = BeautifulSoup(response, features="html.parser")

    return html.find(id="news-table")


def parse_data(data, ticker):
    """
    Parses the data from the news table. Especially the title and date are extracted.
    The values are then appended to an dictionary that contains in each row the ticker,
    date, time and title of the news.

    Args:
        data: The news table data.
        ticker: The ticker.

    Returns:
        The parsed data dictionary.
    """
    parsed_data = []
    for row in data.find_all("tr"):
        title = row.a.get_text()
        date_data = row.td.text.split(" ")

        if len(date_data) == 1:
            time = date_data[0]
        else:
            date = date_data[0]
            time = date_data[1]

        parsed_data.append([ticker, date, time, title])

    return parsed_data


def predict_for_ticker(ticker, start_date, end_date):
    """
    Predicts two sentiment scores for a specified ticker. Basis are news headlines related to the passed ticker
    from the given time range. The news data is exracted from the webiste finviz.com

    Args:
        ticker: The ticker for which the sentiment scores will be calculated.
        start_date: The start date of the news.
        end_date: The end date of the news.

    Returns:
        An dictionary with the parsed news data.
    """
    ticker_data = get_news_table_for_ticker(ticker)
    parsed_ticker_data = parse_data(ticker_data, ticker)

    df = pd.DataFrame(parsed_ticker_data, columns=["ticker", "date", "time", "title"])

    vader = SentimentIntensityAnalyzer()

    encoded_title_function = lambda title: prepare_news_title(title)
    df["encoded_title"] = df["title"].apply(encoded_title_function)

    sentiment_function_model = lambda encoded_title: model.predict(encoded_title)[0]
    df["compound_model"] = df["encoded_title"].apply(sentiment_function_model)

    sentiment_function_vader = lambda title: vader.polarity_scores(title)["compound"]
    df["compound"] = df["title"].apply(sentiment_function_vader)

    # Format time from pm/am to 24h
    df["time"] = (pd.to_datetime(df.time).dt.strftime("%H:%M:%S")).values

    # Combine date and time column
    datetime = pd.to_datetime(df.date + " " + df.time)
    df["date"] = datetime.values

    # Filter for date range
    mask = (df["date"] >= pd.to_datetime(start_date)) & (
        df["date"] <= pd.to_datetime(end_date)
    )
    df = df.loc[mask]

    # Check if data in time range exists
    rows = df.date.count()
    if rows == 0:
        logging.info(
            "Warning: No news data found for the specified time range: '"
            + str(start_date)
            + "' - '"
            + str(end_date)
            + "'"
        )
        return rows, 0, 0.5

    mean_sentiment_vader = df.mean()["compound"]
    mean_sentiment_model = df.mean()["compound_model"]
    logging.info("Used news data entries: " + str(rows))

    return (
        rows,
        mean_sentiment_vader,
        mean_sentiment_model,
    )


def predict_automatically(ticker, hours_till_now):
    """
    Parses the data from the news table. Especially the title and date are extracted.
    The values are then appended to an dictionary that contains in each row the ticker,
    date, time and title of the news.

    Args:
        ticker: The ticker.
        hours_till_now: The time in hours since when we want to search the news.

    Returns:
        Nothing.
    """
    # Get current time and start time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S")
    start_time = current - timedelta(hours=hours_till_now)

    logging.info(
        "Getting prediction for ticker '"
        + ticker
        + "' based on news data of last "
        + str(hours_till_now)
        + " hours ..."
    )
    (
        rows,
        sentiment_score,
        mean_sentiment_model,
    ) = predict_for_ticker(ticker, start_time, current_time)

    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "ticker": ticker,
        "timeframe": {"startTime": start_time, "endTime": current_time},
        "sentimentScoreVader": sentiment_score,
        "sentimentScoreModel": float(mean_sentiment_model),
        "newsCount": int(rows),
        "createdBy": "az-function(news-analysis)",
        "createdAt": current_time,
    }

    return data


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    data = predict_automatically("AMZN", int(os.environ["ANALYSIS_INTERVAL"]))

    return func.HttpResponse(
        json.dumps(data), mimetype="application/json", status_code=200
    )
