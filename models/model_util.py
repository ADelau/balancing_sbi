from .nre import NREFactory
from .bnre import BNREFactory
from .npe import NPEFactory
from .bnpe import BNPEFactory
from .ratio_npe import RatioNPEFactory
from .ratio_bnpe import RatioBNPEFactory
from .arch_initialized_ratio_bnpe import ArchInitializedRatioBNPEFactory
from .nre_c import NRECFactory
from .bnre_c import BNRECFactory

def load_model_factory(config, benchmark, simulation_budget):
    if config["method"] == "nre":
        return NREFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bnre":
        return BNREFactory(config, benchmark, simulation_budget)

    elif config["method"] == "npe":
        return NPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bnpe":
        return BNPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "ratio_npe":
        return RatioNPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "ratio_bnpe":
        return RatioBNPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "arch_initialized_ratio_bnpe":
        return ArchInitializedRatioBNPEFactory(config, benchmark, simulation_budget)

    elif config["method"] == "nre_c":
        return NRECFactory(config, benchmark, simulation_budget)

    elif config["method"] == "bnre_c":
        return BNRECFactory(config, benchmark, simulation_budget)

    else:
        raise NotImplementedError("Model not implemented")