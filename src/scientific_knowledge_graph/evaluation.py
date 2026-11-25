"""Multiclass diagnostics for aligned held-out predictions."""
import numpy as np
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    f1_score, matthews_corrcoef, cohen_kappa_score, roc_auc_score,
    average_precision_score, log_loss, confusion_matrix)

METRIC_KEYS=("accuracy","macro_precision","balanced_accuracy","macro_f1",
    "weighted_f1","mcc","kappa","macro_roc_auc","macro_average_precision",
    "log_loss","brier","ece")

def evaluate_probabilities(y, probabilities, classes, bins=15):
    """Columns must follow the sorted, unique class labels; all classes represented."""
    y=np.asarray(y)
    p=np.asarray(probabilities,dtype=float)
    classes=np.asarray(classes)
    if classes.ndim!=1 or len(classes)<3 or not np.array_equal(classes,np.unique(classes)):
        raise ValueError("Expected at least three sorted unique classes")
    if y.ndim!=1 or not len(y) or p.shape!=(len(y),len(classes)) or not np.isfinite(p).all():
        raise ValueError("Invalid labels or probability dimensions")
    if not np.array_equal(np.unique(y),classes) or np.any(p<0) or np.any(p>1) or not np.allclose(p.sum(axis=1),1,atol=1e-7):
        raise ValueError("Expected valid probabilities and every class in evaluation labels")
    if not isinstance(bins,int) or bins<1: raise ValueError("Invalid bin count")
    truth=(y[:,None]==classes[None,:]).astype(int)
    pred=classes[p.argmax(axis=1)]
    precision,recall,f1,support=precision_recall_fscore_support(y,pred,labels=classes,zero_division=0)
    confidence=p.max(axis=1); correct=pred==y
    bucket=np.minimum((confidence*bins).astype(int),bins-1)
    reliability=[]
    for i in range(bins):
        mask=bucket==i
        if mask.any():
            reliability.append({"bin":i,"count":int(mask.sum()),"confidence":float(confidence[mask].mean()),"accuracy":float(correct[mask].mean())})
    result={"accuracy":float(accuracy_score(y,pred)),
        "macro_precision":float(precision.mean()),"balanced_accuracy":float(recall.mean()),
        "macro_f1":float(f1.mean()),"weighted_f1":float(f1_score(y,pred,average="weighted",zero_division=0)),
        "mcc":float(matthews_corrcoef(y,pred)),"kappa":float(cohen_kappa_score(y,pred)),
        "macro_roc_auc":float(roc_auc_score(truth,p,average="macro")),
        "macro_average_precision":float(average_precision_score(truth,p,average="macro")),
        "log_loss":float(log_loss(y,p,labels=classes)),
        "brier":float(np.mean(np.sum((p-truth)**2,axis=1))),
        "ece":float(sum(b["count"]*abs(b["accuracy"]-b["confidence"]) for b in reliability)/len(y)),
        "classes":classes.tolist(),
        "confusion_matrix":confusion_matrix(y,pred,labels=classes).tolist(),
        "per_class":[{"class":str(c),"precision":float(a),"recall":float(b),"f1":float(d),"support":int(n)} for c,a,b,d,n in zip(classes,precision,recall,f1,support)],
        "reliability_bins":reliability}
    return result,pred

def paired_outcomes(y, reference, candidate):
    """Descriptive paired counts on the same nodes; no independence assumption."""
    y,reference,candidate=map(np.asarray,(y,reference,candidate))
    if y.ndim!=1 or not len(y) or y.shape!=reference.shape or y.shape!=candidate.shape:
        raise ValueError("Aligned nonempty prediction arrays required")
    a=reference==y;b=candidate==y
    return {"both_correct":int((a&b).sum()),"both_wrong":int((~a&~b).sum()),
        "reference_only_correct":int((a&~b).sum()),"candidate_only_correct":int((~a&b).sum()),
        "accuracy_delta":float(b.mean()-a.mean())}
