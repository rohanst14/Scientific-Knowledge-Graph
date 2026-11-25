"""Kernel classification and training-label-only graph probability diffusion."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import normalize

def graph_kernel(features, transition):
    """Combine normalized multi-hop text similarity and diffusion-coordinate similarity."""
    text=normalize(features,norm="l2").astype(np.float64)
    topology=transition
    for _ in range(3):
        topology=topology@transition
    topo=normalize(topology.toarray(),norm="l2").astype(np.float64)
    similarity=.85*(text@text.T)+.15*(topo@topo.T)
    distance=np.maximum(0.,2.-2.*similarity)
    np.fill_diagonal(distance,0.)
    return distance

def diffuse(probabilities, transition, known, known_labels, classes, correction, smoothing, steps=30):
    """Diffuse only residuals and labels from the explicitly supplied training indices."""
    p=np.asarray(probabilities,dtype=float)
    known=np.asarray(known,dtype=int)
    classes=np.asarray(classes)
    target=(np.asarray(known_labels)[:,None]==classes[None,:]).astype(float)
    if len(known)!=len(target) or len(set(known))!=len(known) or not np.all(target.sum(1)==1):
        raise ValueError("Invalid training label alignment")
    if p.shape!=(transition.shape[0],len(classes)) or np.any(known<0) or np.any(known>=len(p)):
        raise ValueError("Invalid graph or training index")
    if not 0<=smoothing<1 or correction<0 or steps<1:
        raise ValueError("Invalid diffusion configuration")
    if correction:
        seed=np.zeros_like(p)
        seed[known]=target-p[known]
        residual=seed.copy()
        for _ in range(steps):
            residual=.8*(transition@residual)+.2*seed
        corrected=np.maximum(p+correction*residual,1e-12)
        corrected/=corrected.sum(1,keepdims=True)
    else:
        corrected=p.copy()
    if not smoothing:
        return corrected
    anchor=corrected.copy()
    anchor[known]=target
    out=anchor.copy()
    for _ in range(steps):
        out=smoothing*(transition@out)+(1-smoothing)*anchor
        out[known]=target
    out=np.maximum(out,1e-12)
    return out/out.sum(1,keepdims=True)

def select_graph_classifier(features, transition, labels, dev, val, train, logistic_c):
    """Search on validation only; refit one selected configuration on the training partition."""
    classes=np.unique(labels[dev])
    if len(set(dev)&set(val)) or not set(dev).union(val)<=set(train):
        raise ValueError("Development and validation must be disjoint training subsets")
    distance=graph_kernel(features,transition)
    bases=[("logistic",float(logistic_c),0.)]+[("kernel_svm",c,gamma) for c in (1.,10.,100.) for gamma in (.5,2.)]
    kernels={g:np.exp(-g*distance) for g in (.5,2.)}
    def fit_predict(config,indices):
        kind,c,gamma=config
        if kind=="logistic":
            model=LogisticRegression(C=c,max_iter=1500,random_state=42)
            model.fit(features[indices],labels[indices])
            p=model.predict_proba(features)
        else:
            kernel=kernels[gamma]
            model=SVC(C=c,kernel="precomputed",probability=True,random_state=42)
            model.fit(kernel[np.ix_(indices,indices)],labels[indices])
            p=model.predict_proba(kernel[:,indices])
        if not np.array_equal(model.classes_,classes):
            raise ValueError("Class order mismatch")
        return p
    candidates=[]
    for config in bases:
        p=fit_predict(config,dev)
        for correction in (0.,1.,3.):
            for smoothing in (0.,.5,.9):
                q=diffuse(p,transition,dev,labels[dev],classes,correction,smoothing)
                pred=classes[q[val].argmax(axis=1)]
                candidates.append({"classifier":config[0],"C":config[1],"gamma":config[2],"correction":correction,"smoothing":smoothing,
                    "validation_accuracy":float(accuracy_score(labels[val],pred)),
                    "validation_macro_f1":float(f1_score(labels[val],pred,average="macro"))})
    best=max(candidates,key=lambda c:(c["validation_accuracy"],c["validation_macro_f1"]))
    p=fit_predict((best["classifier"],best["C"],best["gamma"]),train)
    q=diffuse(p,transition,train,labels[train],classes,best["correction"],best["smoothing"])
    return q,classes,best,candidates
