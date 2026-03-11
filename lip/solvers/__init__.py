from .latino import latino
from .dps import dps
from .mmps import mmps
from .latino_sde import latino_sde
from .lflow import lflow
from .latent_dps import latent_dps
from .latent_latino import latent_latino

ALL = {
    "LATINO": latino,
    "DPS": dps,
    "MMPS": mmps,
    "LATINO+SDE": latino_sde,
    "LFlow": lflow,
}

LATENT_ALL = {
    "Latent LATINO": latent_latino,
    "Latent DPS": latent_dps,
}
