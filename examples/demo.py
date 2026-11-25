"""Controlled-text example, independent of the Cora benchmark."""
import json
from scientific_knowledge_graph import extract

if __name__ == "__main__":
    documents = {"paper-1": "Heat increases pressure. Pressure causes alarm."}
    print(json.dumps(extract(documents), indent=2))
