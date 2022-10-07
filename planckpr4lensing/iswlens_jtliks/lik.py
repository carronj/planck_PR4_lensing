import os, glob
import numpy as np
from copy import deepcopy
import planckpr4lensing
from planckpr4lensing.iswlens_jtliks.utils import camb_clfile, w_pt, w_tt, w_pp

cls_path = os.path.join(os.path.dirname(os.path.abspath(planckpr4lensing.__file__)), 'data_pr4')
cls_unl = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lenspotentialCls.dat'))
CLS_FID = camb_clfile(os.path.join(cls_path, 'FFP10_wdipole_lensedCls.dat'))
CLS_FID['pp'] = np.copy(cls_unl['pp'])
CLS_FID['pt'] = np.copy(cls_unl['pt'])

edges_default = {'pt': np.array([2, 15, 29, 40, 53, 68, 83, 98]),
         'tt': np.arange(2, 100)[::3],
         'pp': np.array([8, 40, 84, 99])}
edges_unbinned = {'tt': np.insert(np.arange(2, 100), -1, 99),
             'pt': np.insert(np.arange(2, 100), -1, 99),
             'pp': np.insert(np.arange(2, 100), -1, 99)}
HL_mtt = False # That's the right way

def oneovercl(cl):
    """Pseudo-inverse for cl-arrays.

    """
    ret = np.zeros_like(cl)
    ret[np.where(cl != 0)] = 1. / cl[np.where(cl != 0)]
    return ret

class jt_lik:
    """

        The many chi2 functions of the instance  takes the Cls (not Dls) as input and fiducial amplitudes)

    """
    def __init__(self, lmax=40, edges_tt=None, edges_pt=None, edges_pp=None, ttpp_cov=0., cmb_key='t', qe_key='p',
                 cobaya_type='tt_pt', typ='npipe'):
        if edges_pp is None:
            edges_pp = edges_unbinned['pp']
        if edges_tt is None:
            edges_tt = edges_unbinned['tt']
        if edges_pt is None:
            edges_pt = edges_unbinned['pt']

        #FIXME: why lmax input?
        self.lmax = lmax
        self.ttpp_cov = ttpp_cov
        self.cmb_key = cmb_key
        self.cobaya_type = cobaya_type
        self.qe_key = qe_key
        self.typ = typ
        self.edges = {'tt':edges_tt, 'pt':edges_pt, 'pp':edges_pp}
        self.lmaxs = {k:self.lmax for k in self.edges.keys()}
        self.calibration_param = 'A_planck'

    def initialize(self):
        assert 0

    def _specsforresp(self):
        specs =[] # spectra keys necessary for 'pt' response calculation
        if self.qe_key in ['ptt', 'p']:
            specs.append('tt')
        if self.qe_key in ['p_p', 'p']: # neglecting bb
            specs.append('ee')
        if self.qe_key in ['p']:
            specs.append('te')
        return specs

    def write(self, folder):
        """Write data on disk. Can then use this to rebin as will

        """
        assert os.path.exists(folder)
        for arr, lab in zip([self.bartt, self.barpt, self.barpp], ['tt', 'pt', 'pp']):
            save_arr = np.zeros(self.lmax + 1)
            edges = self.edges[lab]
            for i, ln in enumerate(arr):
                assert np.sum((ln > 0)) == 1, 'we meant to write only the unbinned thing'
                l = i + int(edges[0])
                assert ln[l] != 0
                save_arr[l] = ln[l]
            np.savetxt(folder + '/binarr_%s.txt'%lab, save_arr)

        np.savetxt(folder + '/meas.txt', self.meas, header='tt tp pp inclusive of bmmc corr')
        np.savetxt(folder + '/Dl_fids.txt', np.concatenate([self.Dtt_fidC, self.Dtp_fidC, self.Dpp_fid]), header='tt tp pp fid BP. not L=0 based ! can get lmin from binning arrays')
        np.savetxt(folder + '/cov.txt', self.cov, header='tt tp pp inclusive of bmmc corr, not L=0 based!')
        np.savetxt(folder + '/covG.txt', self.covG, header='tt tp pp Gaussian trivial corr (no MC correction here), not L=0 based!')
        np.savetxt(folder + '/bmmcs.txt', np.concatenate([self.tt_bmmc, self.tp_bmmc, self.pp_bmmc])) # can get bammc and redo a bmmc from this
        for k in self.dRdlncls.keys():
            np.savetxt(folder + '/dRdlncls_%s.txt'%k, self.dRdlncls[k])
        np.savetxt(folder + '/Rfid.txt', self.Rfid)


    @staticmethod
    def _g_HL(x):
        return np.sign(x - 1.) * np.sqrt(2 * (x - np.log(x) - 1.))

    def _leg(self, meas, pred, HL):
        if HL:
            return self._g_HL(meas / pred)
        else:
            return meas - pred

    def plot_corrmat(self, x=None, y=None, vmin=-0.2, vmax=0.2, cmap='coolwarm'):
        import pylab as pl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig, axmatrix = pl.subplots(1, 1, figsize=(10, 10))
        slix = slice(0, np.sum(list(self.Ns.values()))) if x is None else self.slic[x]
        sliy = slice(0, np.sum(list(self.Ns.values()))) if y is None else self.slic[y]
        s = np.sqrt(np.diag(self.cov))
        cmat = (self.cov / np.outer(s, s))[slix, sliy]
        matrix = pl.imshow(cmat, cmap=cmap, vmin=vmin, vmax=vmax)

        divider = make_axes_locatable(axmatrix)
        cax = divider.append_axes("right", size="3%", pad=0.1)
        pl.colorbar(matrix, cax=cax)
        pl.show()


    def _bpredtt(self, cls):
        ret = np.zeros(self.Ntt)
        #for i, b in enumerate(self.barrs_dict['tt']):
        for i, b in enumerate(self.bartt):
            ret[i] = np.sum(b * w_tt(np.arange(len(b))) * cls['tt'][:len(b)])
        return ret

    def R_Rf(self, cls:dict):
        ret = np.ones(self.lmax + 1, dtype=float)
        if not self.w_pt_resp:
            return ret # lmax of the sky
        dRs = self.dRdlncls # lmax_qlm + 1, lmax_ivf + 1 shaped
        R = np.sum([np.dot(dRs[s], cls[s][:dRs[s].shape[1]] * oneovercl(CLS_FID[s][:dRs[s].shape[1]])) for s in self._specsforresp() ], axis=0)
        n = min(len(R), len(ret))
        ret[:n] *= R[:n] * oneovercl(self.Rfid[:n])
        return ret

    def _bpredpt(self, cls):
        ret = np.zeros(self.Npt)
        R_Rf = self.R_Rf(cls) # Ratio of QE estimator response to fiducial response
        for i, b in enumerate(self.barpt):
            ret[i] = np.sum(b * w_pt(np.arange(len(b))) * cls['pt'][:len(b)] * R_Rf[:len(b)])
        return ret

    def _bpredpp(self, cls):
        ret = np.zeros(self.Npp)
        for i, b in enumerate(self.barpp):
            ret[i] = np.sum(b * w_pp(np.arange(len(b))) * cls['pp'][:len(b)])
        return ret

    def _predtt(self, cltt):
        return cltt[2:self.lmax + 1] * w_tt(np.arange(2, self.lmax + 1)) / self.Dtt_fidC

    def _predtp(self, cltp):
        return cltp[3:self.lmax + 1] * w_pt(np.arange(3, self.lmax + 1)) / self.Dtp_fidC

    def _predpp(self, clpp):
        return clpp[3:self.lmax + 1] * w_pp(np.arange(3, self.lmax + 1)) / self.Dpp_fid

    def get_chi2_tt(self, cls, HL=False, coeffs=None):
        leg = self._leg(self.meas[:self.Ntt], self._bpredtt(cls), HL)
        mat = self.covtt_i
        if coeffs is not None:
            assert coeffs.shape == mat.shape
            d = np.sqrt(np.diag(mat))
            mat = np.outer(d, d) * coeffs
        return np.dot(leg, np.dot(mat, leg))

    def get_chi2_pt(self, cls, HL=False):
        # NB: HL does not make sense in this form for negative ampl.
        leg = self._leg(self.meas[self.Ntt:self.Ntt + self.Npt], self._bpredpt(cls), HL)
        return np.dot(leg, np.dot(self.covtp_i, leg))

    def get_chi2_pp(self, cls, HL=False):
        leg = self._leg(self.meas[self.Ntt + self.Npt:], self._bpredpp(cls), HL)
        return np.dot(leg, np.dot(self.covpp_i, leg))

    def get_chi2_tt_pt(self, cls, subTT=False):
        pred = np.concatenate([self._bpredtt(cls), self._bpredpt(cls)])
        leg = self._leg(self.meas[:self.Ntt + self.Npt], pred, False)
        mat = self.covtt_pt_i
        if subTT:
            mat = self.covtt_pt_i.copy()
            mat[:self.Ntt, :self.Ntt] -= self.covtt_i
        return np.dot(leg, np.dot(mat, leg))

    def get_chi2_tt_pt_mtt(self, cls):
        return self.get_chi2_tt_pt(cls) - self.get_chi2_tt(cls, HL=HL_mtt)

    def get_chi2_tt_pp(self, cls, subTT=False):
        pred = np.concatenate([self._bpredtt(cls), self._bpredpp(cls)])
        leg = self._leg(np.concatenate([self.meas[:self.Ntt], self.meas[self.Ntt + self.Npt:]]), pred, False)
        mat = self.covtt_pp_i
        if subTT:
            mat = self.covtt_pp_i.copy()
            mat[:self.Ntt, :self.Ntt] -= self.covtt_i
        return np.dot(leg, np.dot(mat, leg))

    def get_chi2_tt_pp_mtt(self, cls):
        return self.get_chi2_tt_pp(cls) - self.get_chi2_tt(cls, HL=HL_mtt)

    def get_chi2_pt_pp(self, cls, subPP=False):
        pred = np.concatenate([self._bpredpt(cls), self._bpredpp(cls)])
        leg = self._leg(self.meas[self.Ntt:], pred, False)
        mat = self.covtp_pp_i
        if subPP:
            mat = self.covtp_pp_i.copy()
            mat[self.Npt:, self.Npt:] -= self.covpp_i
        return np.dot(leg, np.dot(mat, leg))

    def get_chi2_tt_pt_pp(self, cls, subTT=False, HL=False, subPP=False):
        pred = np.concatenate([self._bpredtt(cls), self._bpredpt(cls), self._bpredpp(cls)])
        mat = self.covi.copy()
        if subTT:
            mat[:self.Ntt, :self.Ntt] -= self.covtt_i
        if subPP:
            mat[self.Npt + self.Ntt:, self.Npt + self.Ntt:] -= self.covpp_i
        leg = self._leg(self.meas, pred, HL)
        return np.dot(leg, np.dot(mat, leg))

    def get_chi2_tt_pt_pp_mtt(self, cls):
        return self.get_chi2_tt_pt_pp(cls) - self.get_chi2_tt(cls, HL=HL_mtt)

    def get_chi2_tt_pt_pp_mpp(self, cls):
        return self.get_chi2_tt_pt_pp(cls) - self.get_chi2_pp(cls)

    def get_chi2_tt_pt_pp_mtt_mpp(self, cls):
        return self.get_chi2_tt_pt_pp(cls) - self.get_chi2_tt(cls, HL=HL_mtt) - self.get_chi2_pp(cls)


    def get_requirements(self):
        """
         return dictionary specifying quantities calculated by a theory code are needed

         e.g. here we need C_L^{tt} to lmax=2500 and the H0 value
        """
        ret = {'Cl':{}}
        if 'tt' in self.cobaya_type:
            ret['Cl']['tt'] = self.lmax
        if  'pt' in self.cobaya_type or 'tp' in self.cobaya_type:
            ret['Cl']['pt'] = self.lmax
        if self.w_pt_resp: # response calculation
           spec = self._specsforresp()
           for kspec in spec:
               ret['Cl'][kspec] = max(self.lmax, self.dRdlncls[kspec].shape[1] - 1)
        return ret

    def logp(self, **params_values):
        """
        Taking a dictionary of (sampled) nuisance parameter values params_values
        and return a log-likelihood.

        e.g. here we calculate chi^2  using cls['tt'], H0_theory, my_foreground_amp
        """
        # H0_theory = self.provider.get_param("H0")
        # my_foreground_amp = params_values['my_foreground_amp']
        Cls = self.provider.get_Cl(ell_factor=False, units='FIRASmuK2')
        if self.calibration_param in params_values: #FIXME: what to do with PP and PT ? looks like the public lensing lik do nothing
            Cls = deepcopy(Cls) # not sure if this is needed
            for s in ['tt', 'te', 'et', 'ee']:
                if s in Cls.keys():
                    Cls[s] /= params_values[self.calibration_param] ** 2
        chi2 = getattr(self, 'get_chi2_' + self.cobaya_type)(Cls)
        return -0.5 * chi2

class cobaya_jtlik(jt_lik):

    def __init__(self, edges:dict or None=None, cobaya_type='tt', custom_covs=False, custom_resp=False,
                 gauss=False, prefix='jtlik_data_lmax99_PR4_July25', forcePR3TT=False,
                 w_ptbmmc=False, _uniformw=False, cov_ttpp=0., verbose=True):


        self.gauss = gauss # Uses Gaussian analytical covariance if set
        self.custom_covs = custom_covs
        self.custom_resp = custom_resp
        self.rescalcovs = {'PPTT':cov_ttpp}
        self.prefix = prefix
        if prefix in ['jtlik_data_lmax100_PR4_temp', 'jtlik_data_lmax100']:
            _uniformw = True # for comptablity with current chains. These were run with uniform rebinning
            w_ptbmmc = False # same, though we should probbaly keep it this way
        self._uniformw = _uniformw
        self.w_ptbmmc=w_ptbmmc # set bmmc to one if not set (very poorly constrained and small enough anyways)
        self.force_PR3TT = forcePR3TT
        self.verbose = verbose
        self.initialize()
        self.cobaya_type=cobaya_type
        self.calibration_param = 'A_planck'

        if edges is not None:
            self.rebin(edges)
        if verbose:
            print("All these numbers should be perfect ones:")
            print('TT', self._bpredtt(CLS_FID))
            print('PT', self._bpredpt(CLS_FID))
            print('PP', self._bpredpp(CLS_FID))
            print("All these numbers should be resaonable :-)")
            print('TT', self.get_chi2_tt(CLS_FID))
            print('PT', self.get_chi2_pt(CLS_FID))
            print('PP', self.get_chi2_pp(CLS_FID))
            print('TT PT', self.get_chi2_tt_pt(CLS_FID))
            print('TT PT PP', self.get_chi2_tt_pt_pp(CLS_FID))
            print('TT PT mTT', self.get_chi2_tt_pt_mtt(CLS_FID))




    def initialize(self):

        #np.savetxt(folder + '/binarr_%s.txt'%lab, save_arr)  # 0-based
        #np.savetxt(folder + '/meas.txt', self.meas, header='tt tp pp inclusive of bmmc corr')
        #np.savetxt(folder + '/Dl_fids.txt', np.concatenate([self.Dtt_fidC, self.Dtp_fidC, self.Dpp_fid]), header='tt tp pp fid BP. not L=0 based ! can get lmin from binning arrays')
        #np.savetxt(folder + '/cov.txt', self.cov, header='tt tp pp inclusive of bmmc corr, not L=0 based!')
        #np.savetxt(folder + '/covG.txt', self.covG, header='tt tp pp Gaussian trivial corr (no MC correction here), not L=0 based!')
        #np.savetxt(folder + '/bmmcs.txt', np.concatenate([self.tt_bmmc, self.tp_bmmc, self.pp_bmmc])) # can get bammc and redo a bmmc from this

        #----
        self.folder = os.path.join(os.path.dirname(planckpr4lensing.__file__), 'data_pr4','iswliks',self.prefix)
        self.qe_key = 'p'
        #---
        self.cobaya_type = 'tt'
        self.meas  = np.loadtxt(self.folder + '/meas.txt')  # fiducial amplitudes, ~ 1

        bartt_0based = np.loadtxt(self.folder + '/binarr_tt.txt') # 0-based
        barpt_0based = np.loadtxt(self.folder + '/binarr_pt.txt') # 0-based
        barpp_0based = np.loadtxt(self.folder + '/binarr_pp.txt') # 0-based
        Dfids = np.loadtxt(self.folder + '/Dl_fids.txt')
        bmmcs = np.loadtxt(self.folder + '/bmmcs.txt')

        ls_tt = np.arange(np.nonzero(bartt_0based)[0][0], len(bartt_0based))
        ls_pt = np.arange(np.nonzero(barpt_0based)[0][0], len(barpt_0based))
        ls_pp = np.arange(np.nonzero(barpp_0based)[0][0], len(barpp_0based))

        Ntt, Npt, Npp = len(ls_tt), len(ls_pt), len(ls_pp)
        Ntot = Ntt + Npt + Npp
        self.slic ={'tt':slice(0, Ntt), 'pt':slice(Ntt, Ntt + Npt), 'pp':slice(Ntt + Npt, Ntt + Npt + Npp)}

        Ntt_PR3 = 0
        ls_tt_PR3 = np.array([])
        if self.prefix == 'jtlik_data_lmax100':
            barpp_0based *= 1e7  #historical messup
            Dfids[Ntt+Npt:]*= 1e-7  #historical messup

        if self.force_PR3TT: # Use PR3 data points and cov below 30
            ls_tt_PR3 = np.arange(ls_tt[0], 30, dtype=int)
            Ntt_PR3 = ls_tt_PR3.size # Number of TT multipoles to replace with PR3
            folder_PR3 = self.folder.replace(self.prefix,'jtlik_data_lmax99_PR3commander')
            if self.prefix != 'jtlik_data_lmax100':
                if self.verbose:
                    print("ISWLensing: using PR3 TT on %s-%s "%(ls_tt_PR3[0], ls_tt_PR3[-1]))
            CDER_resp = np.loadtxt(folder_PR3 + '/TT_resp.txt')
            fn_PLA = os.path.join(os.path.dirname(planckpr4lensing.__file__), 'data_pr4', 'iswliks', 'COM_PowerSpect_CMB-TT-full_R3.01.txt')
            l_PLA, Dl_PLA, _, _ = np.loadtxt(fn_PLA).transpose()
            l_PLA = np.int_(np.round(l_PLA))
            Dl_fid = CLS_FID['tt'][l_PLA] * l_PLA * (l_PLA + 1.) / (2 * np.pi)
            Al_PLA = np.dot(CDER_resp[:, l_PLA[0]:], (Dl_PLA / Dl_fid)[:199])
            assert (len(Al_PLA) - 1) > ls_tt[-1], 'incompatible lmax'
            bmmcs[:Ntt_PR3] = 1./ np.sum(CDER_resp[:, :119], axis=1)[ls_tt_PR3]
            self.meas[:Ntt_PR3] = Al_PLA[ls_tt_PR3] * bmmcs[:Ntt_PR3]
            #: the following takes the bmmc corrected spectrum, but will be taken out later on if custom resp

            meas_PR3 = np.loadtxt(folder_PR3 + '/meas.txt')
            bmmcs_PR3 = np.loadtxt(folder_PR3 + '/bmmcs.txt')
            assert meas_PR3.size == 3 * (99 - 2 + 1), meas_PR3.size
            assert bmmcs_PR3.size == 3 * (99 - 2 + 1), bmmcs_PR3.size

            self.meas[self.slic['pt'].start:self.slic['pt'].start + Ntt_PR3] = meas_PR3[ (99-2 + 1): (99 - 2 + 1) + Ntt_PR3]
            bmmcs[self.slic['pt'].start:self.slic['pt'].start + Ntt_PR3] = bmmcs_PR3[ (99-2 + 1):(99 - 2 + 1) + Ntt_PR3]
            # not changing pp:
            #self.meas[self.slic['pp'].start:self.slic['pp'].start + Ntt_PR3] = meas_PR3[ 2*(99-2 + 1): 2*(99 - 2 + 1) + Ntt_PR3]
            #bmmcs[self.slic['pp'].start:self.slic['pp'].start + Ntt_PR3] = bmmcs_PR3[ 2*(99-2 + 1): 2*(99 - 2 + 1) + Ntt_PR3]

        lmintt, lmiNpt, lminpp = np.min(ls_tt), np.min(ls_pt), np.min(ls_pp)
        lmaxtt, lmaxtp, lmaxpp = np.max(ls_tt), np.max(ls_pt), np.max(ls_pp)



        covG = np.loadtxt(self.folder + '/covG.txt')
        if not np.allclose(covG, covG.transpose()): # stored upper half only
            for lab1, lab2 in (['tt', 'pt'], ['tt', 'pp'], ['pt', 'pp']):
                covG[self.slic[lab2], self.slic[lab1]] = covG[self.slic[lab1], self.slic[lab2]].transpose()
        if self.gauss:
            cov = covG
        else:
            cov = np.loadtxt(self.folder + '/cov.txt')

        if self.force_PR3TT: # feeding lowL block with commander mask data
            cov_PR3 = np.loadtxt(folder_PR3 + '/cov.txt')
            assert cov_PR3.shape == (3 * (99 - 2 + 1), 3 * (99 - 2 + 1))
            for i1, xy1 in enumerate(['tt', 'pt', 'pp']):
                sli1 = slice(self.slic[xy1].start, self.slic[xy1].start + Ntt_PR3)
                sli1_PR3 = slice(i1 * (99 - 2 + 1), i1 * (99 - 2 + 1) + Ntt_PR3)
                for i2, xy2 in enumerate(['tt', 'pt', 'pp']):
                    if (xy1, xy2) != ('pp', 'pp'):
                        sli2 = slice(self.slic[xy2].start, self.slic[xy2].start + Ntt_PR3)
                        sli2_PR3 = slice(i2 * (99 - 2 + 1), i2 * (99 - 2 + 1) + Ntt_PR3)
                        cov[sli1, sli2] = cov_PR3[sli1_PR3, sli2_PR3]

        if not self.w_ptbmmc: # removing bmmc correction from pt
            bmmc_rescal = np.ones_like(bmmcs)
            bmmc_rescal[self.slic['pt']] = bmmcs[self.slic['pt']]
            self.meas[self.slic['pt']] /= bmmcs[self.slic['pt']]
            cov *= np.outer(1./bmmc_rescal, 1./bmmc_rescal)
            bmmcs[self.slic['pt']] = 1.

        bamcs =  1. / bmmcs - 1. # is this right (unitwise)

        assert len(self.meas) == Ntot, (len(self.meas), Ntot)
        assert cov.shape == (Ntot, Ntot)
        assert lmaxtt == lmaxtp and lmaxpp == lmaxtt
        print("ISWlensing: tt tp pp Nbins ", Ntt, Npt, Npp)
        print("ISWlensing: lmins ", lmintt, lmiNpt, lminpp)
        print("ISWlensing: lmaxs ", lmaxtt, lmaxtp, lmaxpp)

        lmax_sky = 118

        self.lmax = lmax_sky
        self.lmax_rec = min(lmaxtt, lmaxtp, lmaxpp)
        self.lmins = {'tt':ls_tt[0], 'pt':ls_pt[0], 'pp':ls_pp[0]}
        self.lmaxs = {'tt':self.lmax_rec, 'pt':self.lmax_rec, 'pp':self.lmax_rec} # ?
        self.Dfids = Dfids
        self.Dtt_fidC = Dfids[:Ntt]
        self.Dtp_fidC = Dfids[Ntt:Ntt+Npt]
        self.Dpp_fid = Dfids[Ntt+Npt:]
        self.Dfids_dict = {'tt':self.Dtt_fidC, 'pt':self.Dtp_fidC, 'pp':self.Dpp_fid}
        self.bamcs = bamcs
        self.bamcs_dict = {'tt': bamcs[:Ntt], 'pt': bamcs[Ntt:Ntt + Npt], 'pp':bamcs[Ntt+Npt:]}
        self.bmmcs = bmmcs


        self.Ns = {'tt':Ntt, 'pt':Npt, 'pp':Npp}
        self.Ntt, self.Npt, self.Npp = self.Ns['tt'], self.Ns['pt'], self.Ns['pp']


        if self.custom_covs:
            for s in ['TTTT', 'PTTT', 'PTPT', 'PTPP', 'PPTT']:
                fns = [self.folder + '/%s_cov.txt' % s]
                if self.force_PR3TT:  # We must load two matrices and patch them. We start with the base one
                    fns += [self.folder.replace(self.prefix, 'jtlik_data_lmax99_PR3commander') + '/%s_cov.txt' % s]
                for ifn, fn in enumerate(fns): # loads middle and lowL values
    #                fn = self.folder + '/%s_cov.txt'%s # these covariances do not have bmmcs and are L=0-based
                    if os.path.exists(fn):
                        s1, s2 = s[:2].lower(), s[2:].lower()
                        Ns1 = self.Ns[s1] if ifn == 0 else Ntt_PR3
                        Ns2 = self.Ns[s2] if ifn == 0 else Ntt_PR3
                        if ifn == 0:
                            slics1 = self.slic[s1]
                            slics2 = self.slic[s2]
                        else:
                            slics1 = slice(self.slic[s1].start, min(self.slic[s1].stop, self.slic[s1].start + Ns1))
                            slics2 = slice(self.slic[s2].start, min(self.slic[s2].stop, self.slic[s2].start + Ns2))
                        slic1 = slice(self.lmins[s1], self.lmins[s1] + Ns1)
                        slic2 = slice(self.lmins[s2], self.lmins[s2] + Ns2)
                        this_cov = np.loadtxt(fn)[slic1, slic2] * np.outer(self.bmmcs[slics1], self.bmmcs[slics2])
                        if s1 != s2:
                            this_cov = 0.5 * (this_cov + this_cov.T)
                        this_cov *= self.rescalcovs.get(s, 1.)
                        cov[slics1, slics2] = this_cov
                        cov[slics2, slics1] = this_cov.T
                        if self.verbose:
                            print('ISWlensing %s-%s filled in '%(Ns1, Ns2) + s + ' cov from ' + os.path.dirname(fn))

        # reads available responses:
        self.dRdlncls = {}
        fns = glob.glob(self.folder + '/dRdlncls_*.txt')
        for fn in fns:
            spec = os.path.basename(fn).replace('dRdlncls_', '').replace('.txt', '')
            self.dRdlncls[spec] = np.loadtxt(fn)
            if self.verbose:
                print('loaded ' + fn + ' into key ' + spec)


        self.w_pt_resp = False # This will be instantied later


        barrs = np.concatenate([bartt_0based[ls_tt], barpt_0based[ls_pt], barpp_0based[ls_pp]])
        bars_dict = {}
        for s in ['TT', 'PT', 'PP']:
            fns = [self.folder + '/%s_resp.txt'%s]
            if self.force_PR3TT: # We must load two matrices and patch them. We start with the base one
                fns +=  [self.folder.replace(self.prefix, 'jtlik_data_lmax99_PR3commander') + '/%s_resp.txt'%s]
            lmin, lmax_rec = self.lmins[s.lower()], self.lmaxs[s.lower()]
            thisDlfid = (CLS_FID[s.lower()][:lmax_sky + 1] * globals()['w_' + s.lower()](np.arange(lmax_sky + 1)))
            assert np.max(np.abs(thisDlfid[lmin:len(self.Dfids_dict[s.lower()])+lmin] /self.Dfids_dict[s.lower()] - 1.)) < 1e-10
            if np.all([os.path.exists(fn) for fn in fns]) and self.custom_resp:
                bars_dict[s.lower()] = np.zeros((self.Ns[s.lower()], lmax_sky+1))  # otherwise undefined
                for ifn, fn in enumerate(fns): # filling bars_dict array
                    ls_toreplace = ls_tt_PR3 if 'jtlik_data_lmax99_PR3commander' in fn else np.arange( (ls_tt_PR3[-1] + 1) if self.force_PR3TT else lmin, lmax_rec + 1)
                    # We need to: update the binning arrays used in the pred
                    # recalc the bmmcs
                    # update the covs with the new bmmc
                    # barr_{il} -> \sum_l' barr_{il'} D_l' R_{l'l} / D_l
                    # since unbinned  and barr_il = 1/D_l can write barr_il =  R_{i l} / Dl
                    # bamc is <Al> - 1. We have updated the pred, so must calculate <Al> - pred
                    this_resp = np.loadtxt(fn)[:lmax_sky+1, :lmax_sky+1]
                    if 'jtlik_data_lmax99_PR3commander' in fn:
                        bars_dict[s.lower()][ls_toreplace - lmin, :] = this_resp[ls_toreplace, :lmax_sky+1] * oneovercl(thisDlfid)
                    else:
                        bars_dict[s.lower()][ls_toreplace - lmin, :] = this_resp[ls_toreplace, :lmax_sky+1] * oneovercl(thisDlfid) #otherwise undefined
                    if self.verbose:
                        print(" %s-%s adapted binning %s array from "%(ls_toreplace[0], ls_toreplace[-1], s) + os.path.dirname(fn))
                setattr(self, 'bar' + s.lower(), bars_dict[s.lower()]) #hack
                new_pred = getattr(self, '_bpred' + s.lower())(CLS_FID)
                bars_dict[s.lower()] *= np.outer(1./new_pred, np.ones(lmax_sky+1)) # rescaling to get unit cl response to fiducial
                # avA is <D_l> / <D_l^\fid > in the unbinned case
                avA =  1 + self.bamcs_dict[s.lower()]
                newavA =  avA / new_pred
                bmmc = 1./ newavA
                if s.lower() == 'pt' and not self.w_ptbmmc:
                    bmmc = np.ones_like(bmmc)
                    newavA = np.ones_like(newavA)
                self.meas[self.slic[s.lower()]] *= (bmmc /new_pred/ self.bmmcs[self.slic[s.lower()]])
                if self.custom_covs:
                    # custom covs do not have bmmcs in them, while empirical ones do. We also have to include the 1/new_pred making the fiducial prediction unity
                    rescale_bmmc = np.ones_like(self.bmmcs)
                    rescale_bmmc[self.slic[s.lower()]] = bmmc / self.bmmcs[self.slic[s.lower()]] / new_pred
                    cov *= np.outer(rescale_bmmc, rescale_bmmc)
                else:
                    assert 0, 'custom resp without custom cov not implemented here'
                # Need to adapt observed spec, which were given with default bmmc
                self.bmmcs[self.slic[s.lower()]] = np.copy(bmmc)
                self.bamcs[self.slic[s.lower()]] = newavA - 1.
                #assert 0, 'replace bartt etc with dict everywhere .... '

            else:
                lmax_rec = self.lmaxs[s.lower()]
                bar = np.zeros((lmax_rec - lmin + 1, lmax_sky + 1))
                for l in range(lmin, lmax_rec + 1):
                    bar[l - lmin, l] = 1./ thisDlfid[l]
                bars_dict[s.lower()] = np.copy(bar)
        self.barrs = barrs
        self.bartt = bars_dict['tt']
        self.barpt = bars_dict['pt']
        self.barpp = bars_dict['pp']
        self.barrs_dict = bars_dict

        self.cov = cov
        self.covG = covG
        self._build_covi()
        self.binned = False

        self.Rfid = np.loadtxt(self.folder + '/Rfid.txt')
        self.w_pt_resp = np.all([spec in self.dRdlncls.keys() for spec in self._specsforresp()])
        print("ISWlensing: OK for resp calc" * self.w_pt_resp)
        if self.w_pt_resp:
            Rtest = np.sum([np.sum(self.dRdlncls[s], axis=1) for s in self._specsforresp()], axis=0)
            assert np.allclose(self.Rfid[:len(Rtest)], Rtest, rtol=1e-3)
            print('ISWlensing: Rfid test ok')
            self.Rfid = Rtest


    def _build_covi(self):
        Ntt, Npt, Npp = self.Ns['tt'], self.Ns['pt'], self.Ns['pp']
        cov = self.cov
        self.covtt_i = np.linalg.inv(cov[:Ntt, :Ntt])
        self.covtp_i = np.linalg.inv(cov[Ntt:Ntt + Npt, Ntt:Ntt + Npt])
        self.covpp_i = np.linalg.inv(cov[Ntt + Npt:, Ntt + Npt:])
        self.covtt_pt_i = np.linalg.inv(cov[:Ntt + Npt, :Ntt + Npt])
        self.covtp_pp_i = np.linalg.inv(cov[Ntt:, Ntt:])
        self.covi = np.linalg.inv(cov)

        covtt_pp = np.zeros((Ntt + Npp, Ntt + Npp))
        covtt_pp[:Ntt, :Ntt] = cov[:Ntt, :Ntt]
        covtt_pp[Ntt:, Ntt:] = cov[Ntt + Npt:, Ntt + Npt:]
        covtt_pp[:Ntt, Ntt:] = cov[:Ntt, Ntt + Npt:]
        covtt_pp[Ntt:, :Ntt] = cov[Ntt + Npt:, :Ntt]
        self.covtt_pp_i = np.linalg.inv(covtt_pp)

    def _edges2indices(self, spec, edges):
        bls, bus = self._edges2ls(edges)
        retls = [] # just checking lmins and lmaxs
        rtus = []
        lmin = self.lmins[spec]
        lmax = self.lmaxs[spec]
        for bl, bu in zip(bls, bus):
            bl_ = max(lmin, bl)
            bu_ = min(lmax, bu)
            if bu_ >= bl_:
                retls.append(bl_)
                rtus.append(bu_)
        bls = np.array(retls)
        bus = np.array(rtus)
        assert np.all( (bls - lmin) >= 0) and np.all(bus <= lmax) and np.all(bus >= bls) and np.all(np.diff(bls) > 0)
        imin = self.Ns['tt']* (spec not in ['tt']) + self.Ns['pt'] * (spec not in ['tt', 'pt', 'tp'])
        ils = imin + bls - lmin
        ius = imin + bus - lmin

        return ils, ius

    @staticmethod
    def _edges2ls(edges):
        bls = edges[:-1]
        bus = edges[1:] - 1
        bus[-1] += 1
        return bls, bus

    @staticmethod
    def _ls2edges(bls, bus):
        return np.concatenate([bls, [bus[-1]]])

    def corrcoeffs(self, spec1, spec2):
        s = np.sqrt(np.diag(self.cov))
        return self.cov[self.slic[spec1], self.slic[spec2]] / np.outer(s[self.slic[spec1]], s[self.slic[spec2]])

    def rebin_ttcov(self, edges, ttcov):
        ils = np.concatenate([self._edges2indices(k, edges[k])[0] for k in ['tt']])
        ius = np.concatenate([self._edges2indices(k, edges[k])[1] for k in ['tt']])
        Ntot = ils.size
        cov = np.zeros((Ntot, Ntot))
        for i, (il, iu) in enumerate(zip(ils, ius)):
            Ni = (iu - il + 1)
            for j, (jl, ju) in enumerate(zip(ils, ius)):
                Nj = ju - jl + 1
                cov[i, j]= np.sum(self.barrs[il:iu+1] * np.dot(ttcov[il:iu+1, jl:ju+1], self.barrs[jl:ju+1] )) / (Ni * Nj)
        return cov

    def rebin(self, edges:dict):
        assert not self.binned
        ils = np.concatenate([self._edges2indices(k, edges[k])[0] for k in ['tt', 'pt', 'pp']])
        ius = np.concatenate([self._edges2indices(k, edges[k])[1] for k in ['tt', 'pt', 'pp']])
        Ns = {k: len(edges[k]) - 1 for k in edges.keys()}
        Ntot = ils.size
        # new additive MC corr
        # NB: fixed bug here Jun 2022 giving wrong bmmcs, previous lik plots on T0 probably a bit buggy
        #assert np.allclose(np.array([np.mean(self.Dfids[il:iu + 1] * self.barrs[il:iu + 1] ) for il, iu in zip(ils, ius)]), 1., rtol=1e-6)
        Ntt, Npt, Npp = Ns['tt'], Ns['pt'], Ns['pp']

        if self.custom_covs and not self._uniformw:
            print("ISWlensing: rebinning using inverse covariance weighting ", Ntt, Npt, Npp)
            #binwei = 1./np.diag(self.cov)
            binwei = []
            for xy in ['tt', 'pt', 'pp']:
                covi = np.linalg.inv(self.cov[self.slic[xy], self.slic[xy]])
                binwei.append(np.sum(covi, axis=1))
            binwei = np.concatenate(binwei)
        else:
            print("ISWlensing: rebinning with uniform weights ", Ntt, Npt, Npp)
            binwei = np.copy(self.barrs * self.Dfids)  # here this is just uniform weighting !!

        # unit sum weights:
        for b, (il, iu) in enumerate(zip(ils, ius)):
            binwei[il:iu + 1] /= np.sum(binwei[il:iu + 1])
        bamc = np.array([np.sum(self.bamcs[il:iu + 1] * binwei[il:iu+1] ) for il, iu in zip(ils, ius)])
        bmmc = 1. / (1. + bamc)  # new multiplicative correction
        meas = bmmc * np.array([np.sum(self.meas[il:iu + 1] *binwei[il:iu+1] / self.bmmcs[il:iu + 1]) for il, iu in zip(ils, ius)])
        cov_old = self.cov * np.outer(1. / self.bmmcs, 1. /self.bmmcs)
        cov = np.zeros((Ntot, Ntot))
        for i, (il, iu) in enumerate(zip(ils, ius)):
            for j, (jl, ju) in enumerate(zip(ils, ius)):
                cov[i, j]= np.sum(binwei[il:iu+1] * np.dot(cov_old[il:iu+1, jl:ju+1], binwei[jl:ju+1] ))
        cov *= np.outer(bmmc, bmmc)

        self.cov = cov
        self.meas = meas
        self.bmmcs = bmmc
        self.bamcs = bamc
        self.binned = True

        Nls = {k:self.lmaxs[k] - self.lmins[k] + 1 for k in ['tt', 'pt', 'pp']}
        Nltt, Nlpt, Nlpp = Nls['tt'], Nls['pt'], Nls['pp']

        bin_mat = np.zeros((np.sum(list(Ns.values())), np.sum(list(Nls.values()))))
        for b, (il, iu) in enumerate(zip(ils, ius)):
            bin_mat[b, il:iu+1] = binwei[il:iu+1]

        self.bartt = np.dot(bin_mat[:Ntt, :Nltt], self.bartt)
        self.barpt = np.dot(bin_mat[Ntt:Ntt + Npt, Nltt:Nltt+Nlpt], self.barpt)
        self.barpp = np.dot(bin_mat[Ntt + Npt:, Nltt+Nlpt:], self.barpp)

        #bartt = np.zeros((Ns['tt'], self.lmaxs['tt']+ 1))
        #for b, (il, iu) in enumerate(zip(ils[:Ns['tt']], ius[:Ns['tt']])):
        #    self.bartt[b, self.lmins['tt'] + il:self.lmins['tt'] + iu +1] = self.barrs[il: iu+1] / (iu - il + 1)
            #bartt[b, self.lmins['tt'] + il:self.lmins['tt'] + iu +1] = self.barrs_dict['tt'][il: iu+1] / (iu - il + 1)

        #self.barpt = np.zeros((Ns['pt'], self.lmaxs['pt'] + 1))
        #lmin = self.lmins['pt'] - self.Ns['tt']
        #for b, (il, iu) in enumerate(zip(ils[Ns['tt']:Ns['tt']+Ns['pt']], ius[Ns['tt']:Ns['tt']+Ns['pt']])):
        #    self.barpt[b, lmin + il:lmin + iu +1] = self.barrs[il: iu+1] / (iu - il + 1.)

        #self.barpp = np.zeros((Ns['pp'], self.lmaxs['pp'] + 1))
        #lmin = self.lmins['pp'] - self.Ns['tt'] - self.Ns['pt']
        #for b, (il, iu) in enumerate(zip(ils[Ns['tt']+Ns['pt']:], ius[Ns['tt']+Ns['pt']:])):
        #    self.barpp[b, lmin + il:lmin + iu + 1] = self.barrs[il: iu + 1] / (iu - il + 1.)

        self.slic ={'tt':slice(0, Ntt), 'pt':slice(Ntt, Ntt + Npt), 'pp':slice(Ntt + Npt, Ntt + Npt + Npp)}
        self.Ns= Ns
        self.Ntt, self.Npt, self.Npp = self.Ns['tt'], self.Ns['pt'], self.Ns['pp']
        self._build_covi()
        self.edges = edges

class cobaya_jtlikPRXpp(cobaya_jtlik):
    def __init__(self, X=4, prefix='jtlik_data_lmax99_PR4_July25', cobaya_type='tt_pt_pp_mtt', edges=None,
                 force_PR3TT=False, verbose=False):
        """joint TT-PT-PP lik with the published PRX PP likelihood"""
        if str(X) in ['4']: # PR4 release
            from planckpr4lensing import planckpr4lensing
            PRX_PP = planckpr4lensing.PlanckPR4Lensing()
        elif str(X) in ['3']: # PR3 release
            from cobaya.likelihoods.planck_2018_lensing import native
            PRX_PP = native()
        else:
            assert 0, ('dont know what to do with ', X)
        if edges is None:
            edges = {}
        edges_can = {'pt': np.array([2, 10, 20, 29, 40, 53, 68, 84]),
                     'tt': np.concatenate([np.arange(2, 85), [84]]),
                     'pp': np.array([8, 40, 84])}
        for xy in ['pt', 'tt', 'pp']:
            if xy not in edges: edges[xy] = edges_can[xy]

        edges['pp'] = np.array([8, 40, 84])
        ibin, L_min, L_max, L_av, BP, Error, Ahat = PRX_PP.full_bandpowers.T

        self.PRX_PP = PRX_PP
        self.PP_fid = BP / Ahat # These should match the numbers in the papers
        super().__init__(edges=edges, cobaya_type=cobaya_type, custom_covs=True, custom_resp=True,
                   gauss=False, prefix=prefix, w_ptbmmc=False, _uniformw=False, verbose=False, forcePR3TT=force_PR3TT)
        self.lavs = {'pp': L_av}
        for s in ['tt', 'pt']:
            bls, bus =  self._edges2ls(edges[s])
            self.lavs[s] = 0.5 * (bls + bus)
        PP_fid = self._bpredpp(CLS_FID) * self.PP_fid # should be 1e-4 close to previous
        assert np.allclose(PP_fid, self.PP_fid, rtol=1e-3)

        if not np.allclose(np.array([self.meas[-2], self.meas[-1]]), Ahat[:2], rtol=0.02) and verbose:
            print(np.array([self.meas[-2], self.meas[-1]]))
            print(Ahat[:2])
            #assert 0
        Ntt, Npt, Npp = self.Ns['tt'], self.Ns['pt'], self.Ns['pp']
        Npp = Npp + self.PP_fid.size - 2

        cov = np.zeros( (Ntt + Npt + Npp, Ntt + Npt + Npp), dtype=float)
        cov[:self.cov.shape[0], :self.cov.shape[1]] = np.copy(self.cov)
        cov[self.cov.shape[0]-2:, self.cov.shape[1]-2:] = PRX_PP.cov / np.outer(self.PP_fid, self.PP_fid)

        self.cov = cov
        self.meas = np.concatenate([self.meas[:Ntt + Npt], Ahat])

        self.slic ={'tt':slice(0, Ntt), 'pt':slice(Ntt, Ntt + Npt), 'pp':slice(Ntt + Npt, Ntt + Npt + Npp)}
        self.Ns['pp'] = Npp
        self.Ntt, self.Npt, self.Npp = self.Ns['tt'], self.Ns['pt'], self.Ns['pp']
        self._build_covi()
        bins_l_PP = np.array([8, 41, 85, 130, 175, 220, 265, 310, 355])
        bins_u_PP = np.array([40, 84, 129, 174, 219, 264, 309, 354, 400])
        self.edges = {'tt':edges['tt'], 'pt':edges['pt'], 'pp':self._ls2edges(bins_l_PP, bins_u_PP)}
        if verbose:
            print("All these numbers should be perfect ones:")
            print('TT', self._bpredtt(CLS_FID))
            print('PT', self._bpredpt(CLS_FID))
            print('PP', self._bpredpp(CLS_FID))
            print("All these numbers should be resaonable :-)")
            print('TT', self.get_chi2_tt(CLS_FID))
            print('PT', self.get_chi2_pt(CLS_FID))
            print('PP', self.get_chi2_pp(CLS_FID))
            print('TT PT', self.get_chi2_tt_pt(CLS_FID))
            print('TT PT PP', self.get_chi2_tt_pt_pp(CLS_FID))
            print('TT PT mTT', self.get_chi2_tt_pt_mtt(CLS_FID))
            print('TT PT PP mTT', self.get_chi2_tt_pt_pp_mtt(CLS_FID))

    def _cls2dls(self, cls):
        ws = {'TT': w_tt, 'ET': w_tt, 'PP': w_pp, 'EE': w_tt, 'PT': w_pt}
        lab = lambda s: s.lower() if s != 'ET' else 'te'
        req = self.PRX_PP.get_requirements()['Cl']
        dls = {lab(s): ws[s](np.arange(req[s] + 1)) * cls.get(s.lower(), cls.get(s.lower()[::-1], None))[:req[s] + 1]
               for s in list(req.keys())}
        return dls

    def _bpredpp(self, cls):
        self.PRX_PP.get_theory_map_cls(self._cls2dls(cls))
        d = self.PRX_PP.get_binned_map_cls(self.PRX_PP.map_cls).squeeze()
        # lik of PR4 is (d - PP) covi (d- PP). We just renormalize
        return d / self.PP_fid

    def get_requirements(self):
        req = self.PRX_PP.get_requirements() # 2500 for all CMB spectra
        req['Cl']['PT'] = self.lmaxs['pt']
        return req

    def get_chi2_PRXpp(self, cls, HL=False): # Should match get_chi2_pp exactly
        assert not HL
        return -2. * self.PRX_PP.log_likelihood(self._cls2dls(cls))