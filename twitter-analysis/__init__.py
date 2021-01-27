import logging
import json

import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info("Python HTTP trigger function processed a request.")

    data = {"ticker": "AMZN", "Type": "Twitter"}

    return func.HttpResponse(
        json.dumps(data), mimetype="application/json", status_code=200
    )
