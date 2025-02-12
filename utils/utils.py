import jax.numpy as jnp
import jax

def pad_stack(args):
    maxSize = jnp.max(jnp.asarray([arg.size for arg in args]))
    return jnp.stack([jnp.pad(arg,pad_width=(0,maxSize-arg.size),mode="constant",constant_values=-1) for arg in args])

def pad_stack_tree(tree):
    return jax.tree.map(lambda *args: pad_stack(args),*tree)