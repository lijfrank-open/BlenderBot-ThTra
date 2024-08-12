import json
import multiprocessing as mp
import random
import tqdm
from collections import Counter

random.seed(13)


def _norm(x):
    return ' '.join(x.strip().split())


strategies = json.load(open('./strategy.json'))
strategies = [e[1:-1] for e in strategies]
strat2id = {strat: i for i, strat in enumerate(strategies)}
original = json.load(open('./ESConv.json'))


def process_data_esc(d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    dialog_id = d['situation']
    

    d = d['dialog']
    
    result=[]
    i=0
    while d[i]['speaker'] == 'supporter' and i<len(d)-1:
        i+=1
    
    while i<len(d):
        j=i+2
        while j<len(d) and 'feedback' not in d[j]['annotation']:
            j+=1
        if j< len(d) and 'feedback' in d[j]['annotation']:
        
            context =[]
            response =[]
            responses =[]
            score = int(d[j]['annotation']['feedback'])
            con = _norm(d[i]['content'])
          
            if i<j:
                i+=1
            while d[i]['speaker'] == 'seeker':
                con +=" "
                con += _norm(d[i]['content'])
               
                i+=1
            res = _norm(d[i]['content'])
           
            if i<j:
                i+=1
            while  d[i]['speaker'] == 'supporter':
                # print("1s")
                # print(i)
                # print(d[i]['content'])
                res +=" "
                res += _norm(d[i]['content'])
                i+=1
            context.append(con)
            response.append(res)
            k=1
            while k<6:
                if k<=score:
                    response.append("Simple Reflection")
                else:
                    response.append("Warn")
                k+=1
            responses.append(response)
            res1={
                'emotion_type': emotion,
                'problem_type': problem,
                'dialog_id': dialog_id,
                'context':context,
                'responses':responses,
            }
            result.append(res1)
            if i<j:
                context =[]
                response =[]
                responses =[]
                con = _norm(d[i]['content'])
              
                i+=1
                while d[i]['speaker'] == 'seeker':
                    con +=" "
                    con += _norm(d[i]['content'])
                    
                    i+=1
                res = _norm(d[i]['content'])
               
                i+=1
                while  i<len(d) and d[i]['speaker'] == 'supporter':
                    res +=" "
                    res += _norm(d[i]['content'])
                    
                    i+=1
                context.append(con)
                response.append(res)
                k=1
                while k<6:
                    if k<=score:
                        response.append("Simple Reflection")
                    else:
                        response.append("Warn")
                    k+=1
                responses.append(response)

                res2={
                    'emotion_type': emotion,
                    'problem_type': problem,
                    'dialog_id': dialog_id,
                    'context':context,
                    'responses':responses,
                }
                result.append(res2)
        else:
            i = len(d)
    return result   

           

def process_data(d):
    emotion = d['emotion_type']
    problem = d["problem_type"]
    situation = d['situation']
    # init_intensity = int(d['score']['speaker']['begin_intensity'])
    # final_intensity = int(d['score']['speaker']['end_intensity'])

    d = d['dialog']
    dial = []
    for uttr in d:
        text = _norm(uttr['content'])
        role = uttr['speaker']
        if role == 'seeker':
            dial.append({
                'text': text,
                'speaker': 'usr',
            })
        else:
            dial.append({
                'text': text,
                'speaker': 'sys',
                'strategy': uttr['annotation']['strategy'],
            })
    res = {
        'emotion_type': emotion,
        'problem_type': problem,
        'situation': situation,
     
    }
    return res
        
            





data = []

# with mp.Pool(processes=mp.cpu_count()) as pool:
with mp.Pool(processes=1) as pool:
    for e in pool.imap(process_data_esc, tqdm.tqdm(original, total=len(original))):
        data.append(e)


random.seed(13)
random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
with open('./esc_data.txt', 'w') as f:
    for e in train:
        f.write(json.dumps(e) + '\n')


data = []

with mp.Pool(processes=mp.cpu_count()) as pool:
    for e in pool.imap(process_data, tqdm.tqdm(original, total=len(original))):
        data.append(e)

emotions = Counter([e['emotion_type'] for e in data])
problems = Counter([e['problem_type'] for e in data])
print('emotion', emotions)
print('problem', problems)
random.seed(13)
random.shuffle(data)
dev_size = int(0.1 * len(data))
test_size = int(0.1 * len(data))
valid = data[:dev_size]
test = data[dev_size: dev_size + test_size]
train = data[dev_size + test_size:]

print('train', len(train))
with open('./train.txt', 'w') as f:
    for e in train:
        f.write(json.dumps(e) + '\n')

print('valid', len(valid))
with open('./valid.txt', 'w') as f:
    for e in valid:
        f.write(json.dumps(e) + '\n')

print('test', len(test))
with open('./test.txt', 'w') as f:
    for e in test:
        f.write(json.dumps(e) + '\n')
