import sys
import numpy as np
import pandas as pd
import requests
import json
from datetime import datetime
import time

def main(repo_name):
    response = requests.get("http://api.github.com/repos/{}".format(repo_name))
    json_obj = json.loads(response.text)

    date_obj = datetime.strptime(json_obj['created_at'], '%Y-%m-%dT%H:%M:%SZ')
    unix_time = time.mktime(date_obj.timetuple())

    print(unix_time)

if __name__ == '__main__':
    main(sys.argv[1])
