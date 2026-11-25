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
    print("Cora report integrity checks passed")
if __name__=="__main__":check(sys.argv[1])
