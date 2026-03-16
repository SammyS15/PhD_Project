from .dmap import dmap
from .dps import dps
from .dps_cse import dps_cse
from .latent_latino import latent_latino
from .mmps import mmps
from .oracle_langevin import oracle_langevin
from .psld import psld

SOLVERS = {
    "Oracle Langevin": oracle_langevin,
    "Oracle CSE": dps_cse,
    "DMAP": dmap,
    "DPS": dps,
    "MMPS": mmps,
    "PSLD": psld,
    "Latent LATINO": latent_latino,
}
