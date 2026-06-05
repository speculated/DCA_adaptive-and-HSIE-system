import re
import random
from collections import OrderedDict
from pprint import pprint
import pickle as pkl
import json
doc2type = pkl.load(open('data/doc2type.pkl', 'rb'))
entity2type = pkl.load(open('data/entity2type.pkl', 'rb'))
mtype2id = {'PER':0, 'ORG':1, 'GPE':2, 'UNK':3}

def judge(s1, s2):
    if s1==s2:
        return True
    if s2.replace('. ', ' ').replace('.', ' ') == s1:
        return True
    if s2.replace('-', ' ') == s1:
        return True
    return False

def read_csv_file(path):
    data = {}
    flag = 0

    if path.find('aida')>=0:
        flag = 1
    else:
        types = json.load(open('data/generated/type/'+path.split('/')[-1].split('.')[0]+'.json', 'rb'))

    docid = '0'
    with open(path, 'r', encoding='utf8') as f:
        for i, line in enumerate(f):
            comps = line.strip().split('\t')
            doc_name = comps[0] + ' ' + comps[1]
            mention = comps[2]
            mtype = [0,0,0,0]
            if flag == 1:
                doc = ''
                for c in doc_name:
                    try:
                        doc += str(int(c))
                    except:
                        break
                if not doc==docid:
                    docid = doc
                    p = 0
                    # type of tt:[['entity', 'type']]
                    tt = doc2type[docid]
                try:
                    # find type of mention(if no ture, deal recurrently)
                    while not judge(mention.lower(), tt[p][0].lower()):
                        p += 1
                    # mtype2id = {'PER':0, 'ORG':1, 'GPE':2, 'UNK':3}
                    # if can't find type of mention will implement except
                    mtype[mtype2id[tt[p][1]]] = 1

                except:
                    print('no find type of mention: ' + docid+mention) # 370United News of India
                    mtype[mtype2id['UNK']] = 1
            else:
                # type in type.json is related with mention in test.csv
                if path.find('wikipedia')<0:
                    tt = types['sample_%d'%i]['pred'] + types['sample_%d'%i]['overlap']
                    for t in tt:
                        if t == 'MISC':
                            t = 'UNK'
                        if t == 'LOC':
                            t = 'GPE'
                        mtype[mtype2id[t]] = 1
                else:
                    # wikipedia: type in type.json is no related with mention in test.csv
                    mtype[mtype2id['UNK']] = 1

            # context of mention
            lctx = comps[3]
            rctx = comps[4]

            # mention have candicate
            if comps[6] != 'EMPTYCAND':
                cands = [c.split(',') for c in comps[6:-2]]
                cands = [[','.join(c[2:]).replace('"', '%22').replace(' ', '_'), float(c[1])] for c in cands]
            else:
                cands = []

            # get gold entity of mention
            gold = comps[-1].split(',')
            if gold[0] == '-1':
                gold = (','.join(gold[2:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)
            else:
                gold = (','.join(gold[3:]).replace('"', '%22').replace(' ', '_'), 1e-5, -1)

            # --dict data{doc_name: [{mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float())), gold: tuple(gold)}]}
            if doc_name not in data:
                data[doc_name] = []
            data[doc_name].append({'mention': mention,
                                   'mtype': mtype,
                                   'context': (lctx, rctx),
                                   'candidates': cands,
                                   'gold': gold})
    return data


def load_person_names(path):
    data = []
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            data.append(line.strip().replace(' ', '_'))
    # a set(): no order without repetition
    return set(data)

# list represent one doc; dict represent different mention in same doc
# --dict data{doc_name: [{mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float())), gold: tuple(gold)}]}
# person_names: a set()_no order without repetition{str(person)}
def with_coref(dataset, person_names):
    for doc_name, content in dataset.items():
        # content is multiple dict of doc_name in list
        for cur_m in content:
            # cur_m: one dict
            # -- list coref: mention have cur_m in other dict in one list
            coref = find_coref(cur_m, content, person_names)
            if coref is not None and len(coref) > 0:
                cur_cands = {}
                for m in coref:
                    for c, p in m['candidates']:
                        # function of get: return the value of key, p is float
                        cur_cands[c] = cur_cands.get(c, 0) + p
                # --dict cur_cands: {str(candidate): float(p(e|m)))}
                for c in cur_cands.keys():
                    # p(e|m) of candidates divide len(coref)
                    cur_cands[c] /= len(coref)
                # sorted descend by float(p(e|m)
                cur_m['candidates'] = sorted(list(cur_cands.items()), key=lambda x: x[1])[::-1]

    # --dict data{doc_name: [{mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float(), list(etype))), gold: tuple(gold)}]}
    for data_name, content in dataset.items():
        for cur_m in content:
            for i, cand in enumerate(cur_m['candidates']):
                cur_m['candidates'][i] = list(cand)
                cur_m['candidates'][i].append([0, 0, 0, 0])
                if cur_m['candidates'][i][0] in entity2type and len(entity2type[cur_m['candidates'][i][0]]) > 0:
                    for t in entity2type[cand[0]]:
                        cur_m['candidates'][i][-1][mtype2id[t]] = 1
                else:
                    cur_m['candidates'][i][-1][-1] = 1
            for cand in cur_m['candidates']:
                # judge is true
                assert len(cand) == 3

# ment: one dict
# mentlist: multiple dict in one list
def find_coref(ment, mentlist, person_names):
    cur_m = ment['mention'].lower()
    coref = []
    # found mention have part of cur_m, the part of head and the part of tail is same or the head and tail of mention both have one blank
    for m in mentlist:
        # have not candidates or first candidate not in person_names
        if len(m['candidates']) == 0 or m['candidates'][0][0] not in person_names:
            continue

        mention = m['mention'].lower()
        start_pos = mention.find(cur_m)
        if start_pos == -1 or mention == cur_m:
            continue

        end_pos = start_pos + len(cur_m) - 1
        if (start_pos == 0 or mention[start_pos-1] == ' ') and \
                (end_pos == len(mention) - 1 or mention[end_pos + 1] == ' '):
            coref.append(m)

    return coref


def read_conll_file(data, path):
    # --dict conll:{docname: {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]}}
    conll = {}
    with open(path, 'r', encoding='utf8') as f:
        cur_sent = None
        cur_doc = None

        for line in f:
            line = line.strip()
            if line.startswith('-DOCSTART-'):
                # split line according to blank
                docname = line.split()[1][1:]
                conll[docname] = {'sentences': [], 'mentions': []}
                cur_doc = conll[docname]
                cur_sent = []

            else:
                if line == '':
                    cur_doc['sentences'].append(cur_sent)
                    cur_sent = []

                else:
                    comps = line.split('\t')
                    tok = comps[0]
                    cur_sent.append(tok)

                    if len(comps) >= 6:
                        # bi represent tag of NER
                        bi = comps[1]
                        wikilink = comps[4]
                        if bi == 'I':
                            cur_doc['mentions'][-1]['end'] += 1
                        else:
                            new_ment = {'sent_id': len(cur_doc['sentences']),
                                        'start': len(cur_sent) - 1,
                                        'end': len(cur_sent), # the string[] represent [)
                                        'wikilink': wikilink}
                            # --list cur_doc['mentions']:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]
                            cur_doc['mentions'].append(new_ment)

    # merge with data
    # --dict data{doc_name: [{mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float(), list(etype))), gold: tuple(gold)}]}
    # --dict conll:{docname: {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]}}
    # [\W]: special char(no letter, no number,, no chinese, no _)
    rmpunc = re.compile('[\W]+')
    for doc_name, content in data.items():
        conll_doc = conll[doc_name.split()[0]]
        # add {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]} of conll[docname] to [..] of data[doc_name]
        content[0]['conll_doc'] = conll_doc

        cur_conll_m_id = 0
        for m in content:
            mention = m['mention']
            # flag = 0

            while True:
                cur_conll_m = conll_doc['mentions'][cur_conll_m_id]
                # mention is dealed with ' '.join in conll
                cur_conll_mention = ' '.join(conll_doc['sentences'][cur_conll_m['sent_id']][cur_conll_m['start']:cur_conll_m['end']])
                if rmpunc.sub('', cur_conll_mention.lower()) == rmpunc.sub('', mention.lower()):
                    # add {sent_id: int, start: int, end: int, wikilink: str(wikilink)} conll[docname][mentions][cur_conll_m_id] to {..} of data[doc_name][]{..}
                    m['conll_m'] = cur_conll_m


                    # if flag == 1:
                    #     print(cur_conll_m_id, cur_conll_mention, mention)
                    # flag = 0

                    cur_conll_m_id += 1
                    break
                else:
                    # print(cur_conll_m_id, cur_conll_mention, mention)
                    # flag = 1
                    # if no cur_conll_mention == mention
                    cur_conll_m_id += 1

# --dict data{doc_name: [ { mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float())), gold: tuple(gold), conll_m: {sent_id: int, start: int, end: int, wikilink: str(wikilink)} }, conll_doc: {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]} ]}
def reorder_dataset(data, order):
    # the default order is "offset"
    if order == "random" or order == "size":
        for doc_name, content in data.items():
            conll_doc = content[0]['conll_doc']

            if order == "random":
                random.shuffle(data[doc_name])
            elif order == "size":
                data[doc_name] = sorted(data[doc_name], key=lambda x: len(x['candidates']))

            data[doc_name][0]['conll_doc'] = conll_doc


def curriculum_reorder(data):
    sorted_by_value = sorted(data.items(), key=lambda kv: len(kv[1]))

    data_ordered = OrderedDict()
    for doc_name, content in sorted_by_value:
        data_ordered[doc_name] = content

    return data_ordered


def eval(testset, system_pred):
    gold = []
    pred = []

    for doc_name, content in testset.items():
        gold += [c['gold'][0] for c in content]
        pred += [c['pred'][0] for c in system_pred[doc_name]]

    true_pos = 0
    for g, p in zip(gold, pred):
        if g == p and p != 'NIL':
            true_pos += 1

    precision = true_pos / len([p for p in pred if p != 'NIL'])
    recall = true_pos / len(gold)
    f1 = 2 * precision * recall / (precision + recall)
    return f1


class CoNLLDataset:
    """
    reading dataset from CoNLL dataset, extracted by https://github.com/dalab/deep-ed/
    """

    def __init__(self, path, conll_path, person_path, order, method):
        print('load csv and return data(type is dict)')
        # --dict data{doc_name: [{mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float())), gold: tuple(gold)}]}
        # list represent same doc; dict represent different mention in same doc
        self.train = read_csv_file(path + '/aida_train.csv')
        self.testA = read_csv_file(path + '/aida_testA.csv')
        self.testB = read_csv_file(path + '/aida_testB.csv')
        self.msnbc = read_csv_file(path + '/wned-msnbc.csv')
        self.ace2004 = read_csv_file(path + '/wned-ace2004.csv')
        self.aquaint = read_csv_file(path + '/wned-aquaint.csv')
        self.clueweb = read_csv_file(path + '/wned-clueweb.csv')
        self.wikipedia = read_csv_file(path + '/wned-wikipedia.csv')
        self.wikipedia.pop('Jiří_Třanovský Jiří_Třanovský', None)

        print('process coref')
        # a set(): no order without repetition
        person_names = load_person_names(person_path)
        # sorted descend by float(p(e|m), the condition is that mention have cur_m in other dict in one list
        with_coref(self.train, person_names)
        with_coref(self.testA, person_names)
        with_coref(self.testB, person_names)
        with_coref(self.msnbc, person_names)
        with_coref(self.ace2004, person_names)
        with_coref(self.aquaint, person_names)
        with_coref(self.clueweb, person_names)
        with_coref(self.wikipedia, person_names)

        print('load conll')
        # add {sent_id: int, start: int, end: int, wikilink: str(wikilink)} of conll[docname][mentions][cur_conll_m_id] to {..} of data[doc_name][]{..}
        # {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]} of conll[docname] to [..] of data[doc_name]
        # --dict data{doc_name: [ { mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float()), list(etype)), gold: tuple(gold), conll_m: {sent_id: int, start: int, end: int, wikilink: str(wikilink)} }, conll_doc: {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]} ]}
        read_conll_file(self.train, conll_path + '/AIDA/aida_train.txt')
        read_conll_file(self.testA, conll_path + '/AIDA/testa_testb_aggregate_original')
        read_conll_file(self.testB, conll_path + '/AIDA/testa_testb_aggregate_original')
        read_conll_file(self.msnbc, conll_path + '/wned-datasets/msnbc/msnbc.conll')
        read_conll_file(self.ace2004, conll_path + '/wned-datasets/ace2004/ace2004.conll')
        read_conll_file(self.aquaint, conll_path + '/wned-datasets/aquaint/aquaint.conll')
        read_conll_file(self.clueweb, conll_path + '/wned-datasets/clueweb/clueweb.conll')
        read_conll_file(self.wikipedia, conll_path + '/wned-datasets/wikipedia/wikipedia.conll')

        # --dict data{doc_name: [ { mention: str(mention), mtype: list(mtype), context: tuple(str(lctx), str(rctx)), candidates: list(list(str(cands), float(), list(etype))), gold: tuple(gold), conll_m: {sent_id: int, start: int, end: int, wikilink: str(wikilink)} }, conll_doc: {sentences:[list()], mentions:[{sent_id: int, start: int, end: int, wikilink: str(wikilink)}]} ]}
        print('reorder mentions within the dataset')
        reorder_dataset(self.train, order)
        reorder_dataset(self.testA, order)
        reorder_dataset(self.testB, order)
        reorder_dataset(self.msnbc, order)
        reorder_dataset(self.ace2004, order)
        reorder_dataset(self.aquaint, order)
        reorder_dataset(self.clueweb, order)
        reorder_dataset(self.wikipedia, order)

