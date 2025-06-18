"""
Utility Functions for Handling Pytrees, Computations,Etc
"""
import jax.numpy as jnp
import jax
from jax.typing import ArrayLike
import numpy as np

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
    return jnp.sum(x * w[..., jnp.newaxis],axis=0) / jnp.sum(w[...,jnp.newaxis],axis=0)

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
        none_type = type_guide.copy()
        for type in types:
            none_type.pop(int(type))
        types = [type_guide[int(k)] for k in types]
    else:
        types = [int(k) for k in types]
        none_type = {}
    x_z = {key: jnp.zeros((1, value.shape[-1])) for key, value in x[0].items()}
    y_z = {key: jnp.zeros((1, value.shape[-1])) for key, value in y[0].items()}
    x = {k: v for k, v in zip(types, x)}
    y = {k: v for k, v in zip(types, y)}
    x.update({v:x_z for k,v in none_type.items()})
    y.update({v:y_z for k,v in none_type.items()})
    return x,y

def extrap_pdf(pdf1,pdf2,extrap_dist,pdf2_dist):
    return pdf1 + (pdf1-pdf2)*(extrap_dist/pdf2_dist)

def interp_pdf(pdf,dist):
    return weighted_avg(pdf,1./dist)

class CustomArray():
    '''
    Custom Array class for use in Containers.
    '''
    def __init__(self,size,dtype=jnp.asarray(1.).dtype, default_value=-1):
        '''
        Initializes the CustomArray with a given size.
        '''
        self.data = default_value*jnp.ones((size,1), dtype=dtype)
        self.default_value = default_value
    
    def add_item(self,index,item):
        ind_avail = jnp.where(self.data[index] == self.default_value)[0]

        if ind_avail.size != 0:
            self.data = self.data.at[index,ind_avail[0]].set(item)
        else:
            self.data = jnp.concatenate((self.data, (-jnp.ones_like(self.data[...,0:1])).at[index].set(item)), axis=-1)

    def __array__(self):
        return np.asarray(self.data)
    
    def __jax_array__(self):
        return jnp.asarray(self.data)
    
    def __getitem__(self,idx):
        '''
        Allows indexing into the CustomArray.
        '''
        return self.data[idx]

    def add_items(self, index, items: ArrayLike):
        '''
        Adds multiple items to the CustomArray at the specified index.
        '''
        for item in items:
            self.add_item(index, item)

    def shape(self):
        '''
        Returns the shape of the CustomArray.
        '''
        return self.data.shape
    
    def __repr__(self):
        '''
        Returns a string representation of the CustomArray.
        '''
        return f"CustomArray(size={self.data.shape[0]}, data={self.data})"
    
    def dtype(self):
        '''
        Returns the data type of the CustomArray.
        '''
        return self.data.dtype