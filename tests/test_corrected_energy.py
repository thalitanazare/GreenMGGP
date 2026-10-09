import unittest
import numpy as np
from corrected_energy import rebuild,make_functions
from energy_freerun import examples,nrmse
from corrected_experiments import predict

class CorrectedEnergyTests(unittest.TestCase):
    def test_simulators_match_corrected_validation(self):
        exs=examples()
        for key,genes in [('E1',['u1','y1','mul(u1,y1)']),('E2',['u1','q1(y1)','mul(q2(u1),q1(y1))']),('E3',['u1','q2(y1)','mul(y1,y1)'])]:
            ex=exs[key];model,fitted=rebuild(genes,ex);can,other=make_functions(model,fitted,ex)
            expected,truth=predict(fitted,ex['data']['y_val'],ex['data']['u_val'])
            np.testing.assert_allclose(can(),expected,rtol=1e-9,atol=1e-9)
            np.testing.assert_allclose(other(),expected,rtol=1e-9,atol=1e-9)
            self.assertAlmostEqual(nrmse(truth,can()),nrmse(truth,expected),places=10)
if __name__=='__main__':unittest.main()
