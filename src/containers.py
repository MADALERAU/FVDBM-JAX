"""
containers...
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit
from functools import partial
import sys
sys.path.append(".")
from src.dynamics import Dynamics
from utils.utils import *
from src.elements import *
import jax.tree_util
import numpy as np

@jax.tree_util.register_pytree_node_class
class Container():
    dynamics: Dynamics
    '''
    Generic class for all Containers, such as Cells, Nodes, Faces, etc.
    '''
    def __init__(self,size,dynamics: Dynamics):
        '''
        Initializes the Container with a given size and dynamics
        '''
        self.pdf = jnp.ones((size, dynamics.NUM_QUIVERS), dtype=jnp.float32)
        self.dynamics = dynamics

    # JAX flattening and unflattening functions for Container
    def tree_flatten(self):
        children = (self.pdf,)
        aux_data = {'dynamics': self.dynamics}
        return children, aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.pdf = children[0]
        obj.dynamics = aux_data['dynamics']
        return obj
    
    def __repr__(self):
        return repr(self.__dict__)
    
@jax.tree_util.register_pytree_node_class
class Cells(Container):
    '''
    Class for Cells in the FVDBM solver.
    Inherits from Container and initializes with the dynamics.
    '''
    def __init__(self, size, dynamics: Dynamics):
        super().__init__(size, dynamics)
        # Density and Velocity
        self.rho = jnp.zeros((size, 1), dtype=jnp.float32)
        self.vel = jnp.zeros((size, dynamics.DIM), dtype=jnp.float32)

        self.face_indices = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.face_normals = CustomArray(size, dtype=jnp.bool, default_value=0)
        '''Face Normals (0 = negative, 1 = positive)'''

    def init(self):
        '''
        Initializes Cell container for the FVDBM solver.
        '''
        self.face_indices = jnp.asarray(self.face_indices)
        self.face_normals = jnp.asarray(self.face_normals)

    # Jax flattening and unflattening functions for Cells
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()

        children += (self.rho, self.vel)
        
        aux_data["face_indices"] = self.face_indices
        aux_data["face_normals"] = self.face_normals
        return children, aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = super().tree_unflatten({"dynamics":aux_data["dynamics"]}, children[:1])

        obj.rho = children[1]
        obj.vel = children[2]
        obj.face_indices = aux_data["face_indices"]
        obj.face_normals = aux_data["face_normals"]

        return obj
    
    def calc_macros(self):
        self.rho,self.vel = jax.jit(jax.vmap(self.dynamics.calc_macro))(self.pdf)

class Faces(Container):
    '''
    Class for Faces in the FVDBM solver.
    Inherits from Container and initializes with the dynamics.
    '''
    def __init__(self, size, dynamics: Dynamics):
        super().__init__(size, dynamics)
        self.flux = jnp.zeros((size, dynamics.NUM_QUIVERS), dtype=jnp.float32)
        
        #Static
        self.nodes_index = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.stencil_cells_index = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.stencil_dists = CustomArray(size, dtype=jnp.float32, default_value=-1)
        self.n = jnp.zeros((size, dynamics.DIM), dtype=jnp.float32)
        self.L = jnp.zeros((size, 1), dtype=jnp.float32)

    def init(self):
        '''
        Initializes Face container for the FVDBM solver.
        '''
        self.nodes_index = jnp.asarray(self.nodes_index)
        self.stencil_cells_index = jnp.asarray(self.stencil_cells_index)
        self.stencil_dists = jnp.asarray(self.stencil_dists)

class Nodes(Container):
    '''
    Class for Nodes in the FVDBM solver.
    Inherits from Container and initializes with the dynamics.
    '''
    def __init__(self, size, dynamics: Dynamics):
        super().__init__(size, dynamics)
        # Density and Velocity
        self.rho = jnp.zeros((size, 1), dtype=jnp.float32)
        self.vel = jnp.zeros((size, dynamics.DIM), dtype=jnp.float32)

        self.type = jnp.zeros((size, 1), dtype=jnp.int32)  # Node type (e.g., boundary, internal)
        self.cells_index = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.cell_dists = CustomArray(size, dtype=jnp.float32, default_value=-1)

    def init(self):
        '''
        Initializes Node container for the FVDBM solver.
        '''
        self.cells_index = jnp.asarray(self.cells_index)
        self.cell_dists = jnp.asarray(self.cell_dists)

    