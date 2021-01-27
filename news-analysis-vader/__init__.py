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

# Azure imports
import azure.functions as func
from azure.storage.blob import BlobServiceClient

# Definitions
nltk.download("vader_lexicon")
finviz_url = "https://finviz.com/quote.ashx?t="


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
        return rows, 0

    mean_sentiment_vader = df.mean()["compound"]
    logging.info("Used news data entries: " + str(rows))

    return (rows, mean_sentiment_vader)


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
    (rows, sentiment_score) = predict_for_ticker(ticker, start_time, current_time)

    start_time = start_time.strftime("%Y-%m-%d %H:%M:%S")
    data = {
        "ticker": ticker,
        "timeframe": {"startTime": start_time, "endTime": current_time},
        "sentimentScoreVader": sentiment_score,
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
