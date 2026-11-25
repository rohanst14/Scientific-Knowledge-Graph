"""Controlled-text relation extraction, separate from the citation benchmark."""
import re
from collections import defaultdict, deque

PATTERN = re.compile(
    r"^([\w -]+?)\s+(increases|decreases|causes|inhibits|predicts)\s+([\w -]+?)\s*[.]?$",
    re.I,
)
NEGATION = re.compile(r"\b(?:not|never|no|neither|nor|without|cannot)\b", re.I)


def extract(documents):
    """Extract only supported affirmative sentence patterns; retain source IDs."""
    triples = []
    for source, text in documents.items():
        for sentence in re.split(r"[.!?]\s*", text):
            match = PATTERN.fullmatch(sentence.strip())
            if not match or NEGATION.search(sentence):
                continue
            subject, relation, obj = (part.strip().lower() for part in match.groups())
            triples.append({
                "subject": subject, "relation": relation,
                "object": obj, "source": source,
            })
    return triples


def paths(triples, start, target, max_depth=4):
    """Enumerate distinct simple directed paths within an edge-count bound."""
    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a nonnegative integer")
    adjacency = defaultdict(set)
    for triple in triples:
        adjacency[triple["subject"]].add(triple["object"])
    queue = deque([(start.strip().lower(),)])
    target = target.strip().lower()
    found = []
    while queue:
        path = queue.popleft()
        if path[-1] == target:
            found.append(path)
            continue
        if len(path) - 1 >= max_depth:
            continue
        for neighbor in sorted(adjacency[path[-1]]):
            if neighbor not in path:
                queue.append(path + (neighbor,))
    return found
