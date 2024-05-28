import argparse
import re
import sys
import json
from pathlib import Path

_NON_VUL = ['UNDEFINED', 'NONE', 'BENIGN', 'HARMLESS']

def comment_remover(text):
    def replacer(match):
        s = match.group(0)
        if s.startswith('/'):
            return " " # note: a space and not an empty string
        else:
            return s
    pattern = re.compile(
        r'//.*?$|/\*.*?\*/|\'(?:\\.|[^\\\'])*\'|"(?:\\.|[^\\"])*"',
        re.DOTALL | re.MULTILINE
    )
    return re.sub(pattern, replacer, text)

source = ""
with open(sys.argv[1], 'r') as my_file:
    for i, line in enumerate(my_file):
            source = source + ('%04d %s'%(i+1, line))
print(source)

source = comment_remover(source)
print(source)
# for i, line in enumerate(source):
#     source = source + ('%04d %s'%(i+1, line))
