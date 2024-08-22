import jax
import jax.numpy as jnp

class Key: # Class for sim
    def __init__(self,seed):
        self.key = jax.random.key(seed)

    def __call__(self):
        self.key,ret = jax.random.split(self.key)
        return ret

def print_dict(dict: dict):
    for i in dict:
        print(i+": "+str(dict[i]))