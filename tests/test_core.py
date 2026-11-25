from scientific_knowledge_graph import extract,paths
import unittest
class Tests(unittest.TestCase):
    def test_provenance(self): self.assertEqual(extract({'p':'Heat increases pressure.'})[0]['source'],'p')
    def test_path(self): self.assertEqual(paths(extract({'p':'Heat increases pressure. Pressure causes alarm.'}),'heat','alarm'),[('heat','pressure','alarm')])
    def test_cycle(self): self.assertEqual(paths(extract({'p':'A causes B. B causes A.'}),'a','c'),[])
    def test_negation(self):
        self.assertEqual(extract({"p":"Heat never increases pressure. No heat causes alarm."}),[])
    def test_duplicate_paths(self):
        self.assertEqual(paths(extract({"p":"A causes B. A causes B."}),"a","b"),[("a","b")])
    def test_invalid_depth(self):
        with self.assertRaises(ValueError): paths([],"a","b",-1)
