"""Scientific regression checks for the shared corrected evaluation pipeline."""
import unittest
import numpy as np
from corrected_experiments import CorrectedSO, COMMON, fit, predict, run_one, examples, gene_monomial

class CorrectedPredictorTests(unittest.TestCase):
    def setUp(self):
        self.mg=CorrectedSO(inputs=np.zeros((80,1)),outputs=np.zeros((80,1)),**{**COMMON,'generations':1,'populationSize':10},nTerms=3,maxHeight=2,nDelays=2)
    def model(self,genes):
        return self.mg.element.buildModelFromList(genes)
    def test_largest_output_delay(self):
        rng=np.random.default_rng(2);u=rng.normal(size=100);y=np.zeros(100)
        for k in range(2,100):y[k]=0.6*y[k-2]+0.3*u[k]
        m=self.model(['u1','q1(y1)']);f=fit(m,y,u)
        self.assertEqual(f['lag'],2)
        np.testing.assert_allclose(f['theta'],[0,0.3,0.6],atol=1e-13)
        for horizon in [None,20]:
            yp,yd=predict(f,y,u,horizon);np.testing.assert_allclose(yp,yd,atol=1e-13)
    def test_input_only_no_backshift(self):
        u=np.arange(40,dtype=float)/10;y=1+2*u
        f=fit(self.model(['u1']),y,u)
        self.assertEqual(f['lag'],0)
        yp,yd=predict(f,y,u,20);np.testing.assert_allclose(yp,yd,atol=1e-13)
    def test_delay_products_and_duplicate_invariance(self):
        rng=np.random.default_rng(3);u=rng.uniform(-1,1,120);y=np.zeros(120)
        for k in range(3,120):y[k]=0.2*y[k-3]*u[k-2]+0.4*u[k]
        models=[self.model(['mul(q2(y1), q2(u1))','u1']),self.model(['mul(q2(u1), q2(y1))','u1','u1'])]
        predictions=[]
        for m in models:
            f=fit(m,y,u);yp,yd=predict(f,y,u);np.testing.assert_allclose(yp,yd,atol=1e-13);predictions.append(yp)
        np.testing.assert_allclose(*predictions,atol=1e-13)
    def test_divergence_is_not_zeroed(self):
        f=fit(self.model(['y1']),np.arange(80,dtype=float),np.zeros(80))
        f['coef']=[0.,1e200]
        with np.errstate(all='ignore'):
            yp,_=predict(f,np.ones(80),np.zeros(80))
        self.assertFalse(np.all(np.isfinite(yp)))
    def test_equal_calls_and_determinism(self):
        ex=examples()['E1']
        a=run_one(ex,'E1','Green',3,1.,'equal_calls',3)
        b=run_one(ex,'E1','Green',3,1.,'equal_calls',3)
        self.assertEqual(a['evaluations'],280)
        self.assertEqual(a['delivered'],b['delivered'])
        self.assertEqual(a['trace'],b['trace'])
if __name__=='__main__':unittest.main()
