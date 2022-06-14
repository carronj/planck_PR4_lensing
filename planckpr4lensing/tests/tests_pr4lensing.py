import unittest

camb_params = {
    "ombh2": 0.022274,
    "omch2": 0.11913,
    "cosmomc_theta": 0.01040867,
    "As": 0.2132755716e-8,
    "ns": 0.96597,
    "tau": 0.0639}


class LikeTest(unittest.TestCase):

    def test_indep(self):
        from planckpr4lensing import PlanckPR4LensingMarged
        from planckpr4lensing import PlanckPR4Lensing
        import camb
        lmax = 2500
        opts = camb_params.copy()
        opts['lens_potential_accuracy'] = 1
        opts['lmax'] = lmax
        opts['halofit_version'] = 'mead'
        pars = camb.set_params(**opts)
        results = camb.get_results(pars)
        cls = results.get_total_cls(lmax, CMB_unit='muK')
        cl_dict = {p: cls[:, i] for i, p in enumerate(['tt', 'ee', 'bb', 'te'])}
        cl_dict['pp'] = results.get_lens_potential_cls(lmax)[:, 0]

        self.assertAlmostEqual(-2 *  PlanckPR4LensingMarged().log_likelihood(cl_dict),  8.368, 1)
        self.assertAlmostEqual(-2 *  PlanckPR4Lensing().log_likelihood(cl_dict, A_planck=1.0), 8.705, 1)

        # Pol.-only likelihood
        like = PlanckPR4LensingMarged(
            {'dataset_file': 'data_pr4/PP_consext8_npipe_smicaed_Ponly_kfilt_rdn0cov.dataset'})
        #self.assertAlmostEqual(-2 * like.log_likelihood(cl_dict), 13.5, 1)
        print(-2 * like.log_likelihood(cl_dict))

    def test_cobaya(self):
        from cobaya.model import get_model

        info = {'likelihood': {'planckpr4lensing.PlanckPR4Lensing': None},
                'theory': {'camb': {"extra_args": {"lens_potential_accuracy": 1,
                                                    "halofit_version":'mead'}}},
                'params': camb_params, 'debug': True}
        model = get_model(info)
        chi2 = -2 * model.loglikes({'A_planck': 1.0})[0]
        self.assertAlmostEqual(chi2[0], 8.705, 1)

        for name in ['planckpr4lensing.PlanckPR4LensingMarged']:
            info = {'likelihood': {name: None},
                    'theory': {'camb': {"extra_args": {"lens_potential_accuracy": 1,
                                                       "halofit_version":'mead'}, 'stop_at_error': True}},
                    'params': camb_params, 'stop_at_error': True}
        model = get_model(info)
        chi2 = -2 * model.loglikes({})[0]
        self.assertAlmostEqual(chi2[0], 8.368, 1)


if __name__ == '__main__':
    LikeTest().test_cobaya()
    LikeTest().test_indep()