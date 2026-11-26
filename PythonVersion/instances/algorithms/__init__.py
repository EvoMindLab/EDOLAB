"""Algorithm catalogue; implementations load only when requested."""

from core.registry import algorithm_registry

_ALGORITHMS = {
    "ACFPSOAlgorithm": ("ACFPSO", "instances.algorithms.acfpso.algorithm"),
    "AMLPRSCMAESAlgorithm": ("AMLP_RS_CMA_ES", "instances.algorithms.amlp_rs_cma_es.algorithm"),
    "AMLPRSCMSAESIIAlgorithm": ("AMLP_RS_CMSA_ESII", "instances.algorithms.amlp_rs_cmsa_esii.algorithm"),
    "AMPDEAlgorithm": ("AMPDE", "instances.algorithms.ampde.algorithm"),
    "AMPPSOAlgorithm": ("AMPPSO", "instances.algorithms.amppso.algorithm"),
    "AMPPSOBCAlgorithm": ("AMPPSO_BC", "instances.algorithms.amppso_bc.algorithm"),
    "AMPPSOGIAlgorithm": ("AMPPSO_GI", "instances.algorithms.amppso_gi.algorithm"),
    "AMSOAlgorithm": ("AMSO", "instances.algorithms.amso.algorithm"),
    "AmQSOAlgorithm": ("AmQSO", "instances.algorithms.amqso.algorithm"),
    "APCPSOAlgorithm": ("APCPSO", "instances.algorithms.apcpso.algorithm"),
    "CDEAlgorithm": ("CDE", "instances.algorithms.cde.algorithm"),
    "CESOAlgorithm": ("CESO", "instances.algorithms.ceso.algorithm"),
    "CPSOAlgorithm": ("CPSO", "instances.algorithms.cpso.algorithm"),
    "CPSORAlgorithm": ("CPSOR", "instances.algorithms.cpsor.algorithm"),
    "DPCPSOAlgorithm": ("DPCPSO", "instances.algorithms.dpcpso.algorithm"),
    "DSPSOAlgorithm": ("DSPSO", "instances.algorithms.dspso.algorithm"),
    "DynDEAlgorithm": ("DynDE", "instances.algorithms.dynde.algorithm"),
    "DynPopDEAlgorithm": ("DynPopDE", "instances.algorithms.dynpopde.algorithm"),
    "FTMPSOAlgorithm": ("FTMPSO", "instances.algorithms.ftmpso.algorithm"),
    "HiaDriAnAlgorithm": ("HIA_DRI_AN", "instances.algorithms.hia_dri_an.algorithm"),
    "HmSOAlgorithm": ("HmSO", "instances.algorithms.hmso.algorithm"),
    "IDSPSOAlgorithm": ("IDSPSO", "instances.algorithms.idspso.algorithm"),
    "ImQSOAlgorithm": ("ImQSO", "instances.algorithms.imqso.algorithm"),
    "LRPSOAlgorithm": ("LRPSO", "instances.algorithms.lrpso.algorithm"),
    "MPSOAlgorithm": ("MPSO", "instances.algorithms.mpso.algorithm"),
    "MCMAESAlgorithm": ("mCMAES", "instances.algorithms.mcmaes.algorithm"),
    "MDEAlgorithm": ("mDE", "instances.algorithms.mde.algorithm"),
    "MjDEAlgorithm": ("mjDE", "instances.algorithms.mjde.algorithm"),
    "MQSOAlgorithm": ("mQSO", "instances.algorithms.mqso.algorithm"),
    "PsfNBCAlgorithm": ("psfNBC", "instances.algorithms.psfnbc.algorithm"),
    "PSPSOAlgorithm": ("PSPSO", "instances.algorithms.pspso.algorithm"),
    "RAPSOAlgorithm": ("RAPSO", "instances.algorithms.rapso.algorithm"),
    "RPSOAlgorithm": ("RPSO", "instances.algorithms.rpso.algorithm"),
    "SPMPAlgorithm": ("SPMP", "instances.algorithms.spmp.algorithm"),
    "SPSOAPADAlgorithm": ("SPSO_AP_AD", "instances.algorithms.spso_ap_ad.algorithm"),
    "TMIPSOAlgorithm": ("TMIPSO", "instances.algorithms.tmipso.algorithm"),
}

for _class_name, (_name, _module) in _ALGORITHMS.items():
    algorithm_registry.register_lazy(_name, _module, _class_name)


def __getattr__(name: str):
    if name not in _ALGORITHMS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = algorithm_registry.get(_ALGORITHMS[name][0])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | _ALGORITHMS.keys())


__all__ = sorted(_ALGORITHMS)
