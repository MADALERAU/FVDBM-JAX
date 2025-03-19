import jax.numpy as jnp
import jax

def pad_stack(args):
    """
    Pads and stacks a list of JAX arrays to the same size.
    
    Args:
        args (list of jax.numpy.ndarray): List of JAX arrays to be padded and stacked.
    
    Returns:
        jax.numpy.ndarray: A stacked array where all elements are padded to the maximum size.
    """
    maxSize = jnp.max(jnp.asarray([arg.size for arg in args]))
    return jnp.stack([jnp.pad(arg, pad_width=(0, maxSize - arg.size), mode="constant", constant_values=-1) for arg in args])

def pad_stack_tree(tree):
    """
    Applies pad_stack to each element in a JAX pytree.
    
    Args:
        tree (pytree): A JAX pytree containing arrays of varying sizes.
    
    Returns:
        pytree: A new pytree with each element padded and stacked.
    """
    return jax.tree.map(lambda *args: pad_stack(args), *tree)

def weighted_avg(x, w):
    """
    Computes a weighted average of an array.
    
    Args:
        x (jax.numpy.ndarray): Array of values.
        w (jax.numpy.ndarray): Array of weights.
    
    Returns:
        float: The weighted average of x using weights w.
    """
    return jnp.sum(x * w[..., jnp.newaxis]) / jnp.sum(w)

def extrapolate(x: jax.typing.ArrayLike, d: jax.typing.ArrayLike):
    """
    Extrapolates x using the inverse of d as weights.
    
    Args:
        x (jax.typing.ArrayLike): Array of values.
        d (jax.typing.ArrayLike): Distance values used for weighting.
    
    Returns:
        float: The extrapolated value.
    """
    w = 1. / d
    w = jnp.where(w < 0, 0, w)
    return weighted_avg(x, w)

def ind_dict(x: dict, ind: list[int]):
    """
    Selects a subset of a dictionary using a list of indices.
    
    Args:
        x (dict): Dictionary with array-like values.
        ind (list[int]): List of indices to select.
    
    Returns:
        dict: A dictionary with values indexed by ind.
    """
    return {k: v[ind] for k, v in x.items()}

def split_dict(x: dict, k: str):
    """
    Splits a dictionary based on unique values of a specified key.
    
    Args:
        x (dict): Dictionary containing array-like values.
        k (str): Key used for splitting.
    
    Returns:
        tuple: (List of dictionaries split by key values, unique types)
    """
    types = jnp.unique(x[k])
    return [ind_dict(x, jnp.argwhere(x[k] == type)[0]) for type in types], types

def split_2dict(x: dict, y: dict, k: str):
    """
    Splits two dictionaries based on unique values of a specified key.
    
    Args:
        x (dict): First dictionary with array-like values.
        y (dict): Second dictionary with array-like values.
        k (str): Key used for splitting.
    
    Returns:
        tuple: (Lists of split dictionaries for x and y, unique types)
    """
    types = jnp.unique(x[k])
    return [ind_dict(x, jnp.argwhere(x[k] == type)[..., 0]) for type in types], [ind_dict(y, jnp.argwhere(x[k] == type)[..., 0]) for type in types], types

def splice_split_dict(x: dict, k: str, type_guide=None):
    """
    Splits a dictionary and maps unique types using a type guide if provided.
    
    Args:
        x (dict): Dictionary containing array-like values.
        k (str): Key used for splitting.
        type_guide (optional): Mapping of integer keys to new type labels.
    
    Returns:
        dict: A dictionary with type-guided keys mapped to split subsets.
    """
    x, types = split_dict(x, k)
    if type_guide is not None:
        types = [type_guide[int(k)] for k in types]
    else:
        types = [int(k) for k in types]
    return {k: v for k, v in zip(types, x)}

def splice_split_2dict(x: dict, y: dict, k: str, type_guide=None):
    """
    Splits two dictionaries and maps unique types using a type guide if provided.
    
    Args:
        x (dict): First dictionary containing array-like values.
        y (dict): Second dictionary containing array-like values.
        k (str): Key used for splitting.
        type_guide (optional): Mapping of integer keys to new type labels.
    
    Returns:
        tuple: (Two dictionaries with type-guided keys mapped to split subsets)
    """
    x, y, types = split_2dict(x, y, k)
    if type_guide is not None:
        types = [type_guide[int(k)] for k in types]
    else:
        types = [int(k) for k in types]
    return {k: v for k, v in zip(types, x)}, {k: v for k, v in zip(types, y)}
