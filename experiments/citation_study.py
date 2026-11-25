"""Download bounded public OpenAlex records or use a local JSONL snapshot."""
import argparse,json,os,hashlib,datetime
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen,Request
from scientific_knowledge_graph.citations import build_graph,summarize,load_records

def download(path,query,limit):
    if not 1<=limit<=1000: raise ValueError("limit must be 1–1000")
    cursor="*";records=[]
    while cursor and len(records)<limit:
        params={"search":query,"per_page":min(100,limit-len(records)),"cursor":cursor,"select":"id,title,publication_year,referenced_works"}
        key=os.getenv("OPENALEX_API_KEY")
        if key: params["api_key"]=key
        with urlopen(Request("https://api.openalex.org/works?"+urlencode(params),headers={"User-Agent":"ScientificKnowledgeGraph/0.2"}),timeout=60) as response:
            page=json.load(response)
        rows=page["results"]
        if not rows: break
        records.extend(rows);cursor=page["meta"].get("next_cursor")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records),encoding="utf-8")
    path.with_suffix(".metadata.json").write_text(json.dumps({"source":"OpenAlex","query":query,"limit":limit,"retrieved":datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input",type=Path,default=Path("data/openalex.jsonl"))
    p.add_argument("--download",action="store_true")
    p.add_argument("--query",default="process safety")
    p.add_argument("--limit",type=int,default=300)
    p.add_argument("--output",type=Path,default=Path("results/citations.json"))
    a=p.parse_args()
    if a.download: download(a.input,a.query,a.limit)
    records=load_records(a.input)
    if not records: raise ValueError("Empty corpus")
    graph=build_graph(records)
    report={"dataset_sha256":hashlib.sha256(a.input.read_bytes()).hexdigest(),"source_file":a.input.name,"metrics":summarize(graph),"nodes":[{"id":n,**v} for n,v in graph.nodes(data=True)],"edges":[{"from":u,"to":v,**d} for u,v,d in graph.edges(data=True)]}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(report["metrics"],indent=2))
if __name__=="__main__": main()
