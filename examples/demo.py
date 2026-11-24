from scientific_knowledge_graph import *
import json
print(json.dumps(extract({'paper-1':'Heat increases pressure. Pressure causes alarm.'}),indent=2))
