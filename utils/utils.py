import jax.numpy as jnp
import jax

def pad_stack(args):
    maxSize = jnp.max(jnp.asarray([arg.size for arg in args]))
    return jnp.stack([jnp.pad(arg,pad_width=(0,maxSize-arg.size),mode="constant",constant_values=-1) for arg in args])

def pad_stack_tree(tree):
    return jax.tree.map(lambda *args: pad_stack(args),*tree)

# Finds weighted 
def weighted_avg(x,w):
    return jnp.sum(x*w[...,jnp.newaxis])/jnp.sum(w)

# Extrapolates x using 1/d as weighted average
def extrapolate(x:jax.typing.ArrayLike,d:jax.typing.ArrayLike):
    w = 1./d
    w = jnp.where(w<0,0,w)
    return weighted_avg(x,w)

def ind_dict(x: dict,ind: list[int]):
        return {k: v[ind] for k,v in x.items()}

def split_dict(x: dict,k: str):
    types = jnp.unique(x[k])
    return [ind_dict(x,jnp.argwhere(x[k]==type)[0]) for type in types], types

def split_2dict(x: dict,y:dict,k: str):
    types = jnp.unique(x[k])

    return [ind_dict(x,jnp.argwhere(x[k]==type)[...,0]) for type in types], [ind_dict(y,jnp.argwhere(x[k]==type)[...,0]) for type in types], types

def splice_split_dict(x: dict,k: str,type_guide = None):
    x,types = split_dict(x,k)
    if type_guide!=None:
        types = [type_guide[int(k)] for k in types]
    else:
        types = [int(k) for k in types]
    return {k:v for k,v in zip(types,x)}

def splice_split_2dict(x: dict, y: dict,k: str,type_guide = None):
    x,y,types = split_2dict(x,y,k)
    if type_guide!=None:
        types = [type_guide[int(k)] for k in types]
    else:
        types = [int(k) for k in types]
    return {k:v for k,v in zip(types,x)}, {k:v for k,v in zip(types,y)}