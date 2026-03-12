from .latent_latino import latent_latino
from .oracle_langevin import oracle_langevin

SOLVERS = {
    "Latent LATINO": latent_latino,
    "Oracle Langevin": oracle_langevin,
}
