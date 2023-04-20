from .slcp import SLCP
from .spatialsir import SpatialSIR
from .lotka_volterra import LotkaVolterra
from .weinberg import Weinberg
from .two_moons import TwoMoons

def load_benchmark(config):
    if config["benchmark"] == "slcp":
        return SLCP()
    elif config["benchmark"] == "weinberg":
        return Weinberg()
    elif config["benchmark"] == "spatialsir":
        return SpatialSIR()
    elif config["benchmark"] == "lotka_volterra":
        return LotkaVolterra()
    elif config["benchmark"] == "two_moons":
        return TwoMoons()
    else:
        raise NotImplementedError("Benchmark not implemented")