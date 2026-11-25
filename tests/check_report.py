"""Validate the published Cora report against its internal data contract."""
import json,sys,math
from collections import Counter
from pathlib import Path
def check(path):
    r=json.loads(Path(path).read_text())
    ids=[n["id"] for n in r["nodes"]]
    assert len(ids)==len(set(ids))==r["metrics"]["observed_works"]==2708
    edges=[(e["from"],e["to"]) for e in r["edges"]]
    assert len(edges)==len(set(edges))==r["metrics"]["citation_edges"]==5429
    valid=set(ids)
    assert all(a in valid and b in valid for a,b in edges)
    assert Counter(n["category"] for n in r["nodes"])==r["class_counts"]
    assert r["train_count"]+r["test_count"]==len(ids)
    assert len(r["dataset_sha256"])==64
    for key in ("baseline","graph_features","selected_multihop"):
        m=r[key]
        for metric in ("accuracy","macro_f1"):
            assert math.isfinite(m[metric]) and 0<=m[metric]<=1
        assert abs(m["accuracy"]*r["test_count"]-round(m["accuracy"]*r["test_count"]))<1e-7
    assert r["selected_multihop"]["candidate_count"]==20
    if "robustness" in r:
        assert [s["seed"] for s in r["robustness"]["splits"]]==[17,29,73]
        for row in r["robustness"]["splits"]:
            assert row["train_count"]+row["test_count"]==len(ids)
            for key in ("baseline","graph_features","selected_multihop"):
                assert all(math.isfinite(row[key][m]) and 0<=row[key][m]<=1 for m in ("accuracy","macro_f1"))
    if r.get("evaluation_version")==2:
        from scientific_knowledge_graph.evaluation import METRIC_KEYS
        for key in ("baseline","graph_features","selected_multihop"):
            m=r[key]
            assert all(math.isfinite(m[k]) for k in METRIC_KEYS)
            assert all(0<=m[k]<=1 for k in ("accuracy","macro_precision","balanced_accuracy","macro_f1","weighted_f1","macro_roc_auc","macro_average_precision","ece"))
            assert -1<=m["mcc"]<=1 and -1<=m["kappa"]<=1 and m["log_loss"]>=0 and 0<=m["brier"]<=2
            matrix=m["confusion_matrix"]
            assert len(matrix)==len(m["classes"])==7 and all(len(row)==7 for row in matrix)
            assert sum(map(sum,matrix))==sum(c["support"] for c in m["per_class"])==r["test_count"]
            assert abs(sum(matrix[i][i] for i in range(7))/r["test_count"]-m["accuracy"])<1e-9
            assert all(sum(row)==c["support"] for row,c in zip(matrix,m["per_class"]))
            assert sum(b["count"] for b in m["reliability_bins"])==r["test_count"]
        for k,v in r["paired_comparisons"].items():
            assert sum(v[z] for z in ("both_correct","both_wrong","reference_only_correct","candidate_only_correct"))==r["test_count"]
            assert abs((v["candidate_only_correct"]-v["reference_only_correct"])/r["test_count"]-v["accuracy_delta"])<1e-9
    print("Cora report integrity checks passed")
if __name__=="__main__":check(sys.argv[1])
