from planckpr4lensing.iswlens_jtliks import lik
from cobaya.likelihood import Likelihood

npipe_prefix = 'jtlik_data_lmax99_PR4_July25'
PR3_T, verbose = True, False

class pt_mtt(Likelihood):

    def initialize(self):
        mylik = lik.cobaya_jtlikPRXpp(X=4, prefix=npipe_prefix, cobaya_type='tt_pt_mtt', force_PR3TT=PR3_T, verbose=verbose)
        self.mylik = mylik

    def get_requirements(self):
        return self.mylik.get_requirements()

    def logp(self, **params_values):
        self.mylik.provider = self.provider
        return self.mylik.logp(**params_values)

class pt_mtt_mpp(Likelihood):

    def initialize(self):
        mylik = lik.cobaya_jtlikPRXpp(X=4, prefix=npipe_prefix, cobaya_type='tt_pt_pp_mtt_mpp', force_PR3TT=PR3_T, verbose=verbose)
        self.mylik = mylik

    def get_requirements(self):
        return self.mylik.get_requirements()

    def logp(self, **params_values):
        self.mylik.provider = self.provider
        return self.mylik.logp(**params_values)

class cobaya_jtlik_pp(Likelihood):

    def initialize(self):
        mylik = lik.cobaya_jtlikPRXpp(X=4, prefix=npipe_prefix, cobaya_type='pp', force_PR3TT=PR3_T, verbose=verbose)
        self.mylik = mylik

    def get_requirements(self):
        return self.mylik.get_requirements()

    def logp(self, **params_values):
        self.mylik.provider = self.provider
        return self.mylik.logp(**params_values)