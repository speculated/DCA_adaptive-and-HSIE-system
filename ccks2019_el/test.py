#! -*- coding: utf-8 -*-
# import el
from tqdm import tqdm
import sys
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

test_data = []

# with open('./ccks2019_el/train.json') as f:
#   for l in tqdm(f):
#     _ = json.loads(l)
#     test_data.append(_)

# el_data = []
# el_data = [{'text':''}]
# str = sys.argv[1]
# str1 = '{"text":"目前已完成设计工作，主船体已在坞内合拢成型，正在开展设备安装和舾装等建造工作。"}'
# print(_)
# _ = json.loads(str1)
# el_data.append(_)
# res = el.test(test_data)

# el_data = [{"text":"目前已完成设计工作，主船体已在坞内合拢成型，正在开展设备安装和舾装等建造工作。"}]
# el_data[0]['text'] = "目前已完成设计工作，主船体已在坞内合拢成型，正在开展设备安装和舾装等建造工作。"
# print(el_data)

# print(res)


train_data = []

with open('./ccks2019_el/train.json') as (f):
    for l in tqdm(f):
        _ = json.loads(l)
        train_data.append({
            'text': _['text'],
            'mention_data': [
                (x['mention'], int(x['offset']), x['kb_id'])
                for x in _['mention_data'] if x['kb_id'] != 'NIL'
            ]
        })

print(train_data)