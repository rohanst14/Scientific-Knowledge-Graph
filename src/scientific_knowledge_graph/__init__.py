import re
from collections import defaultdict,deque
PATTERN=re.compile(r'^([\w -]+?)\s+(increases|decreases|causes|inhibits|predicts)\s+([\w -]+?)\s*[.]?$',re.I)
def extract(documents):
    triples=[]
    for source,text in documents.items():
        for sentence in re.split(r'[.!?]\s*',text):
            m=PATTERN.fullmatch(sentence.strip())
            if m: triples.append({'subject':m[1].strip().lower(),'relation':m[2].lower(),'object':m[3].strip().lower(),'source':source})
    return triples
def paths(triples,start,target,max_depth=4):
    adj=defaultdict(list)
    for t in triples: adj[t['subject']].append(t['object'])
    queue=deque([(start.lower(),)]); found=[]
    while queue:
        p=queue.popleft()
        if p[-1]==target.lower(): found.append(p); continue
        if len(p)-1>=max_depth: continue
        for nxt in adj[p[-1]]:
            if nxt not in p: queue.append(p+(nxt,))
    return found
