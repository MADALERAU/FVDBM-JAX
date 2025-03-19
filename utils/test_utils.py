import jax
import jax.numpy as jnp

class Key:
    """
    A class for managing JAX random number generation keys.
    
    Attributes:
        key (jax.random.PRNGKey): The current random key.
    """
    
    def __init__(self, seed):
        """
        Initializes the Key object with a seed.
        
        Args:
            seed (int): The seed for JAX random number generation.
        """
        self.key = jax.random.key(seed)

    def __call__(self):
        """
        Splits the current random key and returns a new key.
        
        Returns:
            jax.random.PRNGKey: A new random key.
        """
        self.key, ret = jax.random.split(self.key)
        return ret

def print_dict(dict: dict):
    """
    Prints the contents of a dictionary in a formatted manner.
    
    Args:
        dict (dict): The dictionary to print.
    """
    for i in dict:
        print(i + ": " + str(dict[i]))
