"""Credential-free Cora citation benchmark; original IDs and edge provenance."""
import argparse,hashlib,json
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
    # Select graph depth and regularization exclusively on a development split.
    dev,val=train_test_split(train,test_size=.2,random_state=42,stratify=labels[train])
    from sklearn.preprocessing import normalize
    xnorm=normalize(x,norm="l2")
    representations={0:xnorm}
    smooth=xnorm
    for depth in range(1,5):
        smooth=norm@smooth
        representations[depth]=np.concatenate([representations[depth-1],normalize(smooth,norm="l2")],axis=1)
    candidates=[]
    for depth,features in representations.items():
        for c in (.1,1.,10.,100.):
            estimator=LogisticRegression(C=c,max_iter=1500,random_state=42).fit(features[dev],labels[dev])
            prediction=estimator.predict(features[val])
            candidates.append((float(f1_score(labels[val],prediction,average="macro")),depth,c))
    best=max(candidates,key=lambda item:(item[0],-item[1],-item[2]))
    _,depth,c=best
    selected=LogisticRegression(C=c,max_iter=1500,random_state=42).fit(representations[depth][train],labels[train])
    pred=selected.predict(representations[depth][test])
    advanced={"accuracy":float(accuracy_score(labels[test],pred)),"macro_f1":float(f1_score(labels[test],pred,average="macro")),"depth":depth,"C":c,"validation_macro_f1":best[0],"validation_count":len(val),"candidate_count":len(candidates)}
    def fit(features):
        model=LogisticRegression(max_iter=1000,random_state=42).fit(features[train],labels[train])
        pred=model.predict(features[test])
        return {"accuracy":float(accuracy_score(labels[test],pred)),"macro_f1":float(f1_score(labels[test],pred,average="macro"))}
    # Fixed-config sensitivity study on additional stratified splits.
    # These splits overlap; standard deviations describe variability, not confidence intervals.
    robustness=[]
    for seed in (17,29,73):
        tr,te=train_test_split(np.arange(len(ids)),test_size=.3,random_state=seed,stratify=labels)
        row={"seed":seed,"train_count":len(tr),"test_count":len(te)}
        for key,features,penalty in (("baseline",x,1.),("graph_features",z,1.),("selected_multihop",representations[depth],c)):
            est=LogisticRegression(C=penalty,max_iter=1500,random_state=seed).fit(features[tr],labels[tr])
            pr=est.predict(features[te])
            row[key]={"accuracy":float(accuracy_score(labels[te],pr)),"macro_f1":float(f1_score(labels[te],pr,average="macro"))}
        robustness.append(row)
    summary={key:{metric:{"mean":float(np.mean([r[key][metric] for r in robustness])),"sd":float(np.std([r[key][metric] for r in robustness],ddof=1))} for metric in ("accuracy","macro_f1")} for key in ("baseline","graph_features","selected_multihop")}
    report={"dataset":"Cora (LINQS)","dataset_sha256":hashlib.sha256(raw).hexdigest(),"source_url":URL,"seed":42,"train_count":len(train),"test_count":len(test),"baseline":fit(x),"graph_features":fit(z),"selected_multihop":advanced,"robustness":{"design":"Fixed configuration selected on seed 42; overlapping splits are sensitivity checks, not independent external tests.","splits":robustness,"summary":summary},"metrics":{"observed_works":len(ids),"external_reference_nodes":0,"citation_edges":g.number_of_edges(),"internal_edges":g.number_of_edges(),"weak_components":nx.number_weakly_connected_components(g),"top_pagerank":[{"id":k,"title":"Cora paper "+k,"score":v} for k,v in sorted(rank.items(),key=lambda t:(-t[1],t[0]))[:20]]},"nodes":[{"id":k,"title":"Cora paper "+k,"category":str(y),"observed":True} for k,y in zip(ids,labels)],"edges":[{"from":u,"to":v,"source":"cora.cites","relation":"cites"} for u,v in g.edges],"class_counts":dict(Counter(labels.tolist()))}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps({k:report[k] for k in ("dataset","train_count","test_count","baseline","graph_features","selected_multihop","robustness")},indent=2))
if __name__=="__main__":main()
