import jax.numpy as jnp
import jax

import sys
sys.path.append(".")
from utils.utils import *
from utils.test_utils import *

def pad_stack_test():
    key = Key(1234)

    size1 = jax.random.randint(key(),shape=(),minval=10,maxval=100)
    size2 = jax.random.randint(key(),shape=(),minval=10,maxval=100)

    arr1 = jax.random.normal(key(),(size1,))
    arr2 = jax.random.normal(key(),(size2,))

    list = [arr1,arr2]

    padded = pad_stack(list)

    assert padded.shape == (2,max([size1,size2])), "pad_stack: Output Shape"
    assert jnp.array_equal(padded[0,0:size1],arr1), "pad_stack: Array 1 Equality"
    assert jnp.array_equal(padded[1,0:size2],arr2), "pad_stack: Array 2 Equality"
    
    return True

def extrapolate_test():
    d = jnp.asarray([1,1,2,2,-1,-1])
    x = jnp.asarray([5,10,2,7,0,0])
    extrap = extrapolate(x,d)

    assert extrap == jnp.sum(x[0:4]/d[0:4])/jnp.sum(1/d[0:4])
    return extrap


pad_stack_test()
extrapolate_test()