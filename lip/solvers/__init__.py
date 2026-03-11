from .latino import latino
from .dps import dps
from .mmps import mmps
from .latino_sde import latino_sde
from .lflow import lflow

ALL = {
    "LATINO": latino,
    "DPS": dps,
    "MMPS": mmps,
    "LATINO+SDE": latino_sde,
    "LFlow": lflow,
}
