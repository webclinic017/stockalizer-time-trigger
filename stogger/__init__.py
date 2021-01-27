import datetime
import os
import pymongo
import logging
import requests

import azure.functions as func


def store_data_to_db(put_data):
    myclient = pymongo.MongoClient(os.environ["MongoDBAtlasConnectionString"])
    mydb = myclient["twitter"]
    mycol = mydb["news"]

    x = mycol.insert_one(put_data)


def main(mytimer: func.TimerRequest) -> None:
    utc_timestamp = (
        datetime.datetime.utcnow().replace(tzinfo=datetime.timezone.utc).isoformat()
    )

    if mytimer.past_due:
        logging.info("The timer is past due!")

    # Get news analysis
    logging.info("Stogger retrieving news analysis...")
    response = requests.get(
        "https://live-trading.azurewebsites.net/api/news-analysis-vader?code=XVSYVpckHIBaUz7SFkZZ17OCecW8jmRNoztHKijmq9msVt39CPa05w=="
    )
    data = response.json()

    # TODO: Get twitter analysis
    logging.info("Stogger retrieving twitter analysis...")

    # TODO: Combine analysis
    logging.info("Stogger combining news and twitter analysis...")

    # Retrieving data
    sentiment_score = data["sentimentScoreVader"]
    news_count = int(data["newsCount"])

    if news_count > 0:
        logging.info("News count higher 0")

        # TODO: Sell, hold, buy stocks depending on analysis
        logging.info("Stogger sold/hold/buy stock...")

        # TODO: Save decision to DB (for better understanding)
        logging.info("Stogger saving decision to DB...")
        store_data_to_db(data)
    else:
        logging.info("No News found for analysis - stoping now...")

    logging.info("Stogger function ran at %s", utc_timestamp)
