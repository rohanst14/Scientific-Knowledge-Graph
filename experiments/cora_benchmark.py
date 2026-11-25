"""Credential-free Cora citation benchmark; original IDs and edge provenance."""
import argparse,hashlib,io,json,tarfile
from pathlib import Path
from urllib.request import urlopen
from collections import Counter
import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
URL="https://raw.githubusercontent.com/tkipf/pygcn/master/data/cora/"

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("results/citations.json"));a=p.parse_args()
    with urlopen(URL+"cora.content",timeout=90) as response: content=response.read(20_000_000)
    with urlopen(URL+"cora.cites",timeout=90) as response: cites=response.read(20_000_000)
    raw=content+b"\n"+cites
    records=[line.split() for line in content.decode().splitlines() if line.strip()]
    refs=[line.split() for line in cites.decode().splitlines() if line.strip()]
    ids=[r[0] for r in records];labels=np.array([r[-1] for r in records]);x=np.array([r[1:-1] for r in records],dtype=np.float32)
    if len(ids)!=2708 or len(set(ids))!=2708 or len(refs)!=5429: raise ValueError("Unexpected Cora corpus dimensions")
    g=nx.DiGraph()
    g.add_nodes_from(ids)
    for cited,citing in refs:
        if cited not in g or citing not in g: raise ValueError("Unknown paper ID")
        g.add_edge(citing,cited)
    if set(g.edges)!=set((b,a) for a,b in refs): raise ValueError("Citation integrity failure")
    rank=nx.pagerank(g)
    # Transductive graph features: full observed graph and unlabeled node text, never held-out labels.
    adj=nx.to_scipy_sparse_array(g.to_undirected(),nodelist=ids,dtype=np.float32,format="csr")
    from scipy.sparse import diags,eye
    adj=adj+eye(len(ids),dtype=np.float32,format="csr")
    norm=diags(1/np.asarray(adj.sum(axis=1)).ravel())@adj
    z=np.concatenate([x,norm@x],axis=1)
    train,test=train_test_split(np.arange(len(ids)),test_size=.3,random_state=42,stratify=labels)
    def fit(features):
        model=LogisticRegression(max_iter=1000,random_state=42).fit(features[train],labels[train])
        pred=model.predict(features[test])
        return {"accuracy":float(accuracy_score(labels[test],pred)),"macro_f1":float(f1_score(labels[test],pred,average="macro"))}
    report={"dataset":"Cora (LINQS)","dataset_sha256":hashlib.sha256(raw).hexdigest(),"source_url":URL,"seed":42,"train_count":len(train),"test_count":len(test),"baseline":fit(x),"graph_features":fit(z),"metrics":{"observed_works":len(ids),"external_reference_nodes":0,"citation_edges":g.number_of_edges(),"internal_edges":g.number_of_edges(),"weak_components":nx.number_weakly_connected_components(g),"top_pagerank":[{"id":k,"title":"Cora paper "+k,"score":v} for k,v in sorted(rank.items(),key=lambda t:(-t[1],t[0]))[:20]]},"nodes":[{"id":k,"title":"Cora paper "+k,"category":str(y),"observed":True} for k,y in zip(ids,labels)],"edges":[{"from":u,"to":v,"source":"cora.cites","relation":"cites"} for u,v in g.edges],"class_counts":dict(Counter(labels.tolist()))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:report[k] for k in ("dataset","train_count","test_count","baseline","graph_features")},indent=2))
if __name__=="__main__":main()
