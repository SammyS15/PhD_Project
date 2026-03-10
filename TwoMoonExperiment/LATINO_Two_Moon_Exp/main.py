import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import matplotlib.pyplot as plt
from tarp import get_tarp_coverage

jax.config.update("jax_enable_x64", True)

sigma     = 0.05   # MoG component width (prior)
sigma_min = 1e-2   # VE SDE min noise
sigma_max = 50.0   # VE SDE max noise
sigma_y   = 0.3    # observation noise
N_steps   = 8      # LATINO iterations
N_batch   = 2000   # samples for batch experiment
N_runs    = 500    # LATINO runs for single-obs experiment
SEED      = 0

# Two Moons
resolution = 2000
outer_circ_x = np.cos(np.linspace(0, np.pi, resolution))
outer_circ_y = np.sin(np.linspace(0, np.pi, resolution))
inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, resolution))
inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, resolution)) - 0.5

coords = np.vstack([
    np.append(outer_circ_x, inner_circ_x),
    np.append(outer_circ_y, inner_circ_y),
])

def sample_two_moon_dist(n, rng_key):
    k1, k2 = jr.split(rng_key)
    idx = jr.randint(k1, (n,), 0, coords.shape[1])
    return jnp.array(coords.T)[idx] + sigma * jr.normal(k2, (n, 2))

# USE VE For the Score
def sigma_t(t, sigma_min, sigma_max):
    return sigma_min * (sigma_max / sigma_min) ** t

def ve_score(x, t, sigma_min, sigma_max):
    st = sigma_t(t, sigma_min, sigma_max)
    s2 = sigma ** 2 + st ** 2

    log_w = -0.5 * jnp.sum((x - jnp.array(coords.T) ) ** 2, axis=-1) / s2  
    post_mean = jax.nn.softmax(log_w) @ jnp.array(coords.T)                
    return -(x - post_mean) / s2

def latino(x_init, y_obs, rng_key, N=N_steps):
    """
    LATINO for a single 2D point.

    Args:
        x_init:  shape (2,) -- starting point (use y_obs for standard init)
        y_obs:   shape (2,) -- observation
        rng_key: JAX PRNGKey
        N:       number of LATINO iterations
    Returns:
        x_final: shape (2,)
    """
    timesteps = jnp.linspace(1.0, 1.0 / N, N)  # decreasing: T -> near 0
    keys = jr.split(rng_key, N)

    def step(x, inputs):
        t_k, key = inputs
        sig_k = sigma_t(t_k, sigma_min, sigma_max)

        # Forward diffusion
        x_t = x + sig_k * jr.normal(key, x.shape)

        # Tweedie denoising to replaces LCM
        u_k = x_t + sig_k ** 2 * ve_score(x_t, t_k, sigma_min, sigma_max)

        # 3. Proximal step
        df = jnp.linalg.norm(u_k - y_obs)
        delta_k = sigma_y**2 #jnp.where(t_k > 0.3, 3.0 * df / 10.0, 2.0 * df / 10.0)
        gamma_k = delta_k * sig_k ** 2 / sigma_y ** 2
        x_next = (u_k + gamma_k * y_obs) / (1.0 + gamma_k)
        return x_next, None

    x_final, _ = jax.lax.scan(step, x_init, (timesteps, keys))
    return x_final

latino_batch = jax.jit(jax.vmap(lambda y, key: latino(y, y, key)))

# Run LATINO and check results
key = jr.PRNGKey(SEED)
key, k1, k2, k3 = jr.split(key, 4)

# Ground truth and observations
x_true = sample_two_moon_dist(N_batch, k1)                           
Y_obs  = x_true + sigma_y * jr.normal(k2, x_true.shape)     

# LATINO outputs
batch_keys = jr.split(k3, N_batch)
x_latino = latino_batch(Y_obs, batch_keys)                   

# True posterior samples (one per observation)
key, k4 = jr.split(key)
post_keys = jr.split(k4, N_batch)                                                     

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"LATINO Two Moons Denoising  ($\\sigma_y$={sigma_y}, N={N_steps})", fontsize=14)

xlim, ylim = (-1.5, 2.5), (-1.0, 2.0)
titles = ["True prior $p(x)$", f"Nosiy Samples ($\\sigma_y$={sigma_y})", "LATINO output"]
data   = [x_true, Y_obs, x_latino]

for ax, d, title in zip(axes, data, titles):
    d_np = np.array(d)
    ax.scatter(d_np[:, 0], d_np[:, 1], s=2, alpha=0.4)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title)
    ax.set_xlabel("$x_1$")

axes[0].set_ylabel("$x_2$")
plt.tight_layout()
plt.savefig("./Plots/comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# Run TARP to evaluate the quality of the 'posterior'
def plot_tarp_curve(ecp, alpha, ecp_bootstrap, alpha_bootstrap, sigma_y):
    k_sigma = [1, 2, 3]
    
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.plot([0, 1], [0, 1], ls='--', color='k', label = "Ideal case")
    ax.plot(alpha, ecp_bootstrap.mean(axis=0), label='TARP')
    for k in k_sigma:
        ax.fill_between(alpha, ecp_bootstrap.mean(axis=0) - k * ecp_bootstrap.std(axis=0), ecp_bootstrap.mean(axis=0) + k * ecp_bootstrap.std(axis=0), alpha = 0.2)
    ax.legend()
    ax.set_ylabel("Expected Coverage")
    ax.set_xlabel("Credibility Level")
    out_path = f"./Plots/tarp_curve.png"
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"  Saved TARP plot to {out_path}")

N_tarp_obs  = 200   # number of test observations
N_tarp_post = 100  # LATINO posterior samples per observation

x_true_tarp_list  = []  # will stack to (N_tarp_obs, 2)
x_latino_tarp_list = []  # will stack to (N_tarp_obs, N_tarp_post, 2)

for _ in range(N_tarp_obs):
    key, k_x, k_noise, k_lat = jr.split(key, 4)

    # Sample a new latent and generate an observation
    x_true_i = sample_two_moon_dist(1, k_x)[0]                      # (2,)
    y_obs_i  = x_true_i + sigma_y * jr.normal(k_noise, (2,))        # (2,)

    x_true_tarp_list.append(x_true_i)

    # N_tarp_post independent LATINO runs for this observation
    post_keys_i = jr.split(k_lat, N_tarp_post)
    x_lat_i = jax.vmap(lambda k: latino(y_obs_i, y_obs_i, k))(post_keys_i)  # (N_tarp_post, 2)
    x_latino_tarp_list.append(x_lat_i)

x_true_tarp   = jnp.stack(x_true_tarp_list)    # (N_tarp_obs, 2)
x_latino_tarp = jnp.stack(x_latino_tarp_list)  # (N_tarp_obs, N_tarp_post, 2)
# TARP expects (n_samples, n_sims, n_dims) -> transpose to (N_tarp_post, N_tarp_obs, 2)
x_latino_tarp = jnp.transpose(x_latino_tarp, (1, 0, 2))

ecp, alpha = get_tarp_coverage(
    np.array(x_latino_tarp),  # (n_samples=100, n_sims=50, n_dims=2)
    np.array(x_true_tarp),    # (n_sims=50, n_dims=2)
    references='random', 
    metric='euclidean', 
    norm = True
    )

ecp_bootstrap, alpha_bootstrap = get_tarp_coverage(
    np.array(x_latino_tarp),  # (n_samples=100, n_sims=50, n_dims=2)
    np.array(x_true_tarp),    # (n_sims=50, n_dims=2)
    references="random",
    metric="euclidean",
    norm=True,
    bootstrap=True,
)

plot_tarp_curve(ecp, alpha, ecp_bootstrap, alpha_bootstrap, sigma_y)