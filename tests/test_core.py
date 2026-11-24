from scientific_knowledge_graph import extract,paths
import unittest
class Tests(unittest.TestCase):
    def test_provenance(self): self.assertEqual(extract({'p':'Heat increases pressure.'})[0]['source'],'p')
    def test_path(self): self.assertEqual(paths(extract({'p':'Heat increases pressure. Pressure causes alarm.'}),'heat','alarm'),[('heat','pressure','alarm')])
    def test_cycle(self): self.assertEqual(paths(extract({'p':'A causes B. B causes A.'}),'a','c'),[])
