import jax
import jax.numpy as jnp
from src.fvdbmDynamics import Dynamics
from jax.typing import ArrayLike
from jax import Array, jit
from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Element():

    # Static
    dynamics: Dynamics

    # Init/Jax Methods
    def __init__(self,pdf,rho,vel):
        self.pdf = pdf
        self.rho = rho
        self.vel = vel

    @classmethod
    def ones_init(cls): # for init w/ pdf as ones
        temp = cls.__new__(cls)
        temp.pdf = cls.dynamics.ones_pdf()
        temp.rho = 0
        temp.vel = jnp.asarray([0,0])
        temp.calc_macro()
        return temp

    def tree_flatten(self):
        children = (self.pdf,self.rho,self.vel)
        aux_data = {}
        return(children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)

    def __str__(self):
        return f'Element(Density: {self.rho}, Velocity: {self.vel.squeeze()})'
    
    # Class Methods

    def density(self): # returns calculated density
        return jnp.sum(self.pdf)
    
    def velocity(self): # returns calculated velocity
        return jnp.dot(self.dynamics.KSI.T,self.pdf)/self.rho
    
    def calc_macro(self): # calculates & sets density & velocity
        self.rho = self.density()
        self.vel = self.velocity()

    def equilibrium(self): # returns equilibrium pdf based on current pdf
        return self.dynamics.calc_eq(self.rho,self.vel)
    
    def calc_eq(self): # calculated & sets equilibrium pdf as current pdf
        self.pdf = self.equilibrium()

@register_pytree_node_class
class Cell(Element):

    # Init/Jax Methods
    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike,pdf_eq:ArrayLike,faces_index:ArrayLike):
        super().__init__(pdf,rho,vel)

        # Defining Cell Vars.
        self.pdf_eq = pdf_eq

        # STATIC VARIABLES #
        #define faces in cell # should be an ArrayLike['int']
        self.faces_index = faces_index

    @classmethod
    def eq_init(cls,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike,faces_index:ArrayLike):
        temp = cls.__new__(cls)
        temp.super_init(pdf,rho,vel)
        temp.faces_index = faces_index
        #calculated eq pdf
        temp.pdf_eq = temp.dynamics.ones_pdf()
        temp.pdf_eq = temp.equilibrium()

    def super_init(self,pdf,rho,vel):
        super().__init__(pdf,rho,vel)

    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        children += (self.pdf_eq,)
        aux_data.update({'faces_index': self.faces_index})
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    
    # Class Methods
    def get_neq_pdf(self): # returns non-equilibrium pdf
        return self.pdf-self.pdf_eq
    
    def calc_eq(self,Faces:ArrayLike): # calculates eq using face flux
        pass
    
class Face(Element):

    #Init/Jax Methods   
    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike,
                 flux:ArrayLike,node_index:ArrayLike,stencil_cell_index:ArrayLike,stencil_dist: ArrayLike):
        super().__init__(pdf,rho,vel)
        # Defining Face Vars.

        self.flux = flux

        # STATIC VARIABLES #
        self.node_index = node_index
        self.stencil_cell_index = stencil_cell_index
        self.stencil_dist = stencil_dist

    def tree_flatten(self):
        children,aux_data = super().tree_flatten()
        children += (self.flux,)
        aux_data.update({'node_index':self.node_index,'stencil_cell_index':self.stencil_cells_index,'stencil_dist': self.stencil_dist})
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    
    # Class Methods
    def calc_eq(self): # via stenciling
        pass
    
    def calc_flux(self):
        pass

class Node(Element):

    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike):
        super().__init__(pdf,rho,vel)

        # Defining Node Vars.
        self.cells: Array[Cell] = []
        self.cell_dists: Array[float] = []
    
    