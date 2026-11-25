"""Build a directed citation graph from OpenAlex work records."""
import json,re
from pathlib import Path
import networkx as nx

def work_id(value):
    if not isinstance(value,str) or not re.fullmatch(r"(?:https://openalex\.org/)?W[0-9]+",value):
        raise ValueError("Invalid OpenAlex work ID")
    return value.rsplit("/",1)[-1]

def build_graph(records):
    graph=nx.DiGraph()
    seen=set()
    for record in records:
        identifier=work_id(record["id"])
        if identifier in seen: raise ValueError("Duplicate work: "+identifier)
        seen.add(identifier)
        year=record.get("publication_year")
        graph.add_node(identifier,title=record.get("title") or identifier,year=year,observed=True)
        for reference in record.get("referenced_works",[]):
            target=work_id(reference)
            if target==identifier: continue
            if target not in graph: graph.add_node(target,title=target,year=None,observed=False)
            graph.add_edge(identifier,target,relation="cites",source=record["id"])
    return graph

def summarize(graph):
    observed=[n for n,a in graph.nodes(data=True) if a["observed"]]
    induced=graph.subgraph(observed).copy()
    rank=nx.pagerank(induced) if len(induced) else {}
    return {"observed_works":len(observed),"external_reference_nodes":len(graph)-len(observed),
            "citation_edges":graph.number_of_edges(),"internal_edges":induced.number_of_edges(),
            "weak_components":nx.number_weakly_connected_components(induced),
            "top_pagerank":[{"id":n,"title":graph.nodes[n]["title"],"score":v} for n,v in sorted(rank.items(),key=lambda x:(-x[1],x[0]))[:20]]}

def evidence_path(graph,source,target,cutoff=4):
    if not 1<=cutoff<=8: raise ValueError("cutoff must be 1–8")
    source,target=work_id(source),work_id(target)
    if source not in graph or target not in graph: return []
    try: path=nx.shortest_path(graph,source,target)
    except nx.NetworkXNoPath: return []
    if len(path)-1>cutoff: return []
    return [{"from":a,"to":b,**graph.edges[a,b]} for a,b in zip(path,path[1:])]

def load_records(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]
