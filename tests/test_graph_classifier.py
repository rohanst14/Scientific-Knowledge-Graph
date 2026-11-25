import unittest
import numpy as np
from scipy.sparse import csr_matrix
from scientific_knowledge_graph.graph_classifier import diffuse, graph_kernel

class GraphClassifierTests(unittest.TestCase):
    def test_diffusion(self):
        transition=csr_matrix([[.5,.5,0],[.25,.5,.25],[0,.5,.5]])
        p=np.full((3,3),1/3)
        q=diffuse(p,transition,[0],[2],[0,1,2],1,.8)
        np.testing.assert_allclose(q.sum(1),1)
        self.assertEqual(q[0].argmax(),2)
        self.assertEqual(q[1].argmax(),2)
        self.assertEqual(q[2].argmax(),2)
        np.testing.assert_allclose(p,1/3)
    def test_no_diffusion(self):
        p=np.array([[.2,.3,.5],[.4,.4,.2]])
        q=diffuse(p,csr_matrix(np.eye(2)),[0],[0],[0,1,2],0,0)
        np.testing.assert_allclose(q,p)
    def test_graph_kernel(self):
        d=graph_kernel(np.eye(3),csr_matrix(np.eye(3)))
        np.testing.assert_allclose(d,d.T)
        np.testing.assert_allclose(np.diag(d),0)
        self.assertTrue((d>=0).all())
