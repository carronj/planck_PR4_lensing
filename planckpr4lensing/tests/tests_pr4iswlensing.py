import unittest

camb_params = {
    "ombh2": 0.022274,
    "omch2": 0.11913,
    "cosmomc_theta": 0.01040867,
    "As": 0.2132755716e-8,
    "ns": 0.96597,
    "tau": 0.0639}


class LikeTest(unittest.TestCase):

    def test_cobaya(self):
        from cobaya.model import get_model
        chi2s = []
        for name in ['planckpr4lensing.planckpr4iswlensing.ISWLensingTT', 'planckpr4lensing.planckpr4iswlensing.ISWLensingTTPP']:
            info = {'likelihood': {name: None},
                    'theory': {'camb': {"extra_args": {"lens_potential_accuracy": 1,
                                                       "halofit_version":'mead'}, 'stop_at_error': True}},
                    'params': camb_params, 'stop_at_error': True}
            model = get_model(info)
            chi2s.append(-2 * model.loglikes({})[0])
        #self.assertAlmostEqual(chi2[0], 8.368, 1)

        info = {'likelihood': {'planckpr4lensing.PlanckPR4Lensing': None},
                'theory': {'camb': {"extra_args": {"lens_potential_accuracy": 1,
                                                    "halofit_version":'mead'}}},
                'params': camb_params, 'debug': True}
        model = get_model(info)
        chi2s.append(-2 * model.loglikes({'A_planck': 1.0})[0])
        return chi2s

    def test_indep(self):
        from planckpr4lensing.planckpr4iswlensing import ISWLensingTT, ISWLensingTTPP
        from planckpr4lensing.planckpr4iswlensing import lik
        from copy import deepcopy
        cls_fi = deepcopy(lik.CLS_FID)
        cls_fi['tt'] *= 1.01
        lik_pt_mtt = ISWLensingTT()
        lik_pt_mtt_mpp = ISWLensingTTPP()
        chi2_mtt = lik_pt_mtt.mylik.get_chi2_tt_pt_mtt(cls_fi)
        chi2_mtt_mpp = lik_pt_mtt_mpp.mylik.get_chi2_tt_pt_pp_mtt_mpp(cls_fi)
        chi2_pp = lik_pt_mtt_mpp.mylik.get_chi2_pp(cls_fi)
        chi2_PR4pp = lik_pt_mtt_mpp.mylik.get_chi2_PRXpp(cls_fi)

        self.assertAlmostEqual(chi2_pp, 10.88730806)#9.314049476)
        self.assertAlmostEqual(chi2_pp, chi2_PR4pp)
        self.assertAlmostEqual(chi2_mtt_mpp + chi2_pp,13.1444216)# 11.5816895)
        return  chi2_mtt, chi2_mtt_mpp, chi2_pp
if __name__ == '__main__':
    chi2s = LikeTest().test_indep()
    chi2sb = LikeTest().test_cobaya()
    print(chi2sb)