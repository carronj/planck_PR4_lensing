try:
    # more recent versions
    from cobaya.likelihoods.base_classes import CMBlikes
except ImportError:
    from cobaya.likelihoods._base_classes import _CMBlikes as CMBlikes


class PlanckPR4Lensing(CMBlikes):
    # You can either keep data separate and set install_options for where to download from, or
    # (if not too big) bundle the data in the package, so don't need separate install as here
    install_options = {}


class PlanckPR4LensingMarged(PlanckPR4Lensing):
    pass