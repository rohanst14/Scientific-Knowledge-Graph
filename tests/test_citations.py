import unittest
from scientific_knowledge_graph.citations import build_graph,summarize,evidence_path
class CitationTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{"id":"W1","title":"A","referenced_works":["W2","W3","W2"]},{"id":"W2","title":"B","referenced_works":["W3"]},{"id":"W3","title":"C","referenced_works":["W4","W3"]}]
    def test_counts(self):
        g=build_graph(self.rows);m=summarize(g)
        self.assertEqual((m["observed_works"],m["external_reference_nodes"],m["citation_edges"],m["internal_edges"]),(3,1,4,3))
        self.assertAlmostEqual(sum(r["score"] for r in m["top_pagerank"]),1)
    def test_provenance(self):
        g=build_graph(self.rows)
        self.assertEqual(evidence_path(g,"W1","W4",3)[0]["source"],"W1")
        self.assertEqual(len(evidence_path(g,"W1","W4")),2)
        self.assertEqual(evidence_path(g,"W1","W4",1),[])
        self.assertEqual(evidence_path(g,"W4","W1"),[])
    def test_duplicate(self):
        with self.assertRaises(ValueError): build_graph(self.rows+self.rows)
    def test_invalid(self):
        with self.assertRaises(ValueError): build_graph([{"id":"bad"}])
    def test_empty(self):
        self.assertEqual(summarize(build_graph([]))["observed_works"],0)
