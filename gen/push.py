import os
import json
from datetime import datetime
import logging
import pandas as pd
import urllib.request
from fibberio import Task
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv()

# get endpoint vars
url = os.environ['endpoint']

# headers
headers = {
    'Content-Type':'application/json',
    'Authorization':('Bearer '+ os.environ['key'])
}

def create(count: int):
    task = Task('programmers.json')
    df: pd.DataFrame = task.generate(count)
    return df

def request(age, location, orgzm, style, yoe, projects):
    data =  {
        "programmer": {
            "age": age,
            "location": location,
            "orgsz": orgzm,
            "style": style,
            "yoe": yoe,
            "projects": projects
        }
    }

    body = str.encode(json.dumps(data))
    req = urllib.request.Request(url, body, headers)
    try:
        response = urllib.request.urlopen(req)
        result = json.loads(response.read())
        if result['message'] =="Success!":
            info = f'model({datetime.fromisoformat(result["model_update"]):%Y-%m-%d %H:%M}), pred({result["prediction"]:3s} -> {result["scores"][result["prediction"]]:.0%}), time({result["time"]})'
            logging.info(info)
        else:
            info = f'model({datetime.fromisoformat(result["model_update"]):%Y-%m-%d %H:%M}), execution error: {result["message"]}'
            logging.error(info)
    except urllib.error.HTTPError as error:
        logging.error("The request failed with status code: " + str(error.code))

def main(count: int):
    logging.info("Creating data frame")
    df = create(count)
    logging.info("Starting threads")
    with ThreadPoolExecutor(max_workers=32, thread_name_prefix="Worker") as executor:
        for index, row in df.iterrows():
            executor.submit(request, row['age'], row['location'], row['orgsz'], row['style'], row['yoe'], row['projects'])

if __name__ == '__main__':
    format = "[\033[31;1;1m%(threadName)-9s\033[0m][\033[32;1;1m%(asctime)s\033[0m]: %(message)s"
    logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
    #request(34.0, 'South', 751.0, 'spaces', 19.0, 26.0)
    main(10000)
    

