import argparse
import re
import sys
import json
from pathlib import Path

_NON_VUL = ['UNDEFINED', 'NONE', 'BENIGN', 'HARMLESS']

data = ""
with open(sys.argv[1], 'r') as my_file:
    data = my_file.read()

_JSON_START = "```json"
_JSON_END = "```"
data_str = data[data.find(_JSON_START) + len(_JSON_START) + 1: data.rfind(_JSON_END)]
#print(data_str)

data_dict = json.loads(data_str)
print(data_dict)

svd_dict = [d for d in data_dict if d['vulnerability-type'].upper() not in _NON_VUL]

if not svd_dict:
    print("No vulnerabilities detected!")
else:
    json_object = json.dumps(svd_dict, indent=4)
    json_file = "../logs/" + Path(sys.argv[1]).stem + ".json"
    with open(json_file, "w") as outfile:
        outfile.write(json_object)
