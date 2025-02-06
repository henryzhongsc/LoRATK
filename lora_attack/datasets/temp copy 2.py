import glob
import json

ctba_file = 'backdoor500_negsentiment_ctba.json'
mtba_file = 'backdoor500_negsentiment_mtba.json'

with open(ctba_file, 'r') as f:
    ctba_data = json.load(f)

with open(mtba_file, 'r') as f:
    mtba_data = json.load(f)

for ctba_item, mtba_item in zip(ctba_data, mtba_data):
    mtba_item['answer'] = ctba_item['answer']

with open(mtba_file, 'w') as f:
    json.dump(mtba_data, f, indent=4)

