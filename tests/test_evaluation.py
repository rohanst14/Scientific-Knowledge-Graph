import unittest
import numpy as np
from scientific_knowledge_graph.evaluation import evaluate_probabilities,paired_outcomes,METRIC_KEYS
class EvaluationTests(unittest.TestCase):
    def test_perfect(self):
        m,p=evaluate_probabilities([0,1,2],np.eye(3),[0,1,2])
        self.assertEqual(p.tolist(),[0,1,2])
        for k in METRIC_KEYS[:9]: self.assertAlmostEqual(m[k],1.)
        for k in METRIC_KEYS[9:]: self.assertAlmostEqual(m[k],0.)
        self.assertEqual(m["confusion_matrix"],np.eye(3,dtype=int).tolist())
    def test_constant(self):
        m,_=evaluate_probabilities([0,1,2],[[1,0,0]]*3,[0,1,2])
        self.assertAlmostEqual(m["accuracy"],1/3)
        self.assertAlmostEqual(m["balanced_accuracy"],1/3)
        self.assertAlmostEqual(m["macro_precision"],1/9)
        self.assertAlmostEqual(m["macro_f1"],1/6)
        self.assertAlmostEqual(m["weighted_f1"],1/6)
        self.assertAlmostEqual(m["mcc"],0)
        self.assertAlmostEqual(m["kappa"],0)
        self.assertAlmostEqual(m["macro_roc_auc"],.5)
        self.assertAlmostEqual(m["macro_average_precision"],1/3)
        self.assertAlmostEqual(m["brier"],4/3)
        self.assertAlmostEqual(m["ece"],2/3)
    def test_invalid_probabilities(self):
        for p in ([[.2,.2,.2]]*3,[[float("nan"),0,1]]*3,[[-1,1,1]]*3):
            with self.assertRaises(ValueError): evaluate_probabilities([0,1,2],p,[0,1,2])
    def test_paired(self):
        m=paired_outcomes([0,1,2,0],[0,2,2,1],[1,1,2,2])
        self.assertEqual(m,{"both_correct":1,"both_wrong":1,"reference_only_correct":1,"candidate_only_correct":1,"accuracy_delta":0.})
    def test_missing_class(self):
        with self.assertRaises(ValueError): evaluate_probabilities([0,1],[[1,0,0],[0,1,0]],[0,1,2])
