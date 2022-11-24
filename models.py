# Generalized Lagrangian Networks | 2020
# Miles Cranmer, Sam Greydanus, Stephan Hoyer (...)
import jax
if jax.__version__ == '0.1.68':
    from jax.experimental import stax
else:
    from jax.example_libraries import stax

def mlp(args):
    return stax.serial(
        stax.Dense(args.hidden_dim),
        stax.Softplus,
        stax.Dense(args.hidden_dim),
        stax.Softplus,
        stax.Dense(args.output_dim),
    )

def pixel_encoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_latent_dim),
    )

def pixel_decoder(args):
    return stax.serial(
        stax.Dense(args.ae_hidden_dim),
        stax.Softplus,
        stax.Dense(args.ae_input_dim),
    )
