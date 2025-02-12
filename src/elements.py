import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit
from functools import partial
import sys
sys.path.append(".")
from src.dynamics import Dynamics
from utils.utils import *

class Element():

    # Static
    dynamics: Dynamics

    # Init/Jax Methods
    def __init__(self,pdf,rho,vel):
        self.pdf = pdf
        self.rho = rho
        self.vel = vel

    @classmethod
    def pdf_init(cls,pdf):
        temp = cls.__new__(cls)
        temp.pdf = pdf
        # temp.rho = None
        # temp.vel = None
        temp.calc_macro()
        return temp

    @classmethod
    def ones_init(cls): # for init w/ pdf as ones
        temp = cls.__new__(cls)
        temp.pdf = cls.dynamics.ones_pdf()
        # temp.rho = None
        # temp.vel = None
        temp.calc_macro()
        return temp

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children = (self.pdf,self.rho,self.vel)
        aux_data = {}
        return(children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    ### END ###

    def __str__(self):
        return f'{(type(self).__name__)}(Density: {self.rho}, Velocity: {self.vel.squeeze()})'
    
    # Class Methods

    @partial(jax.jit, static_argnums = 0)
    def density(self,pdf): # returns calculated density
        return jnp.sum(pdf)
    
    @partial(jax.jit, static_argnums = 0)
    def velocity(self,pdf,rho): # returns calculated velocity
        return jnp.dot(self.dynamics.KSI.T,pdf)/rho
    
    def equilibrium(self,rho,vel): # returns equilibrium pdf based on current pdf
        return self.dynamics.calc_eq(rho,vel)
    
    def calc_macro(self): # calculates & sets density & velocity
        self.rho = self.density(self.pdf)
        self.vel = self.velocity(self.pdf,self.rho)
    
    def calc_eq(self): # calculated & sets equilibrium pdf as current pdf
        self.pdf = self.equilibrium(self.rho,self.vel)

class ElementContainer:
    def __init__(self,elements: list[Element]):
        self.elements = elements

    def flatten(self):
        assert all(self.elements[0].__class__ == element.__class__ for element in self.elements), "ElementContainer: Element types differ"
        assert len(self.elements) > 0, "ElementContainer: Empty List"
        return pad_stack_tree([element.__dict__ for element in self.elements])

class Cell(Element):

    # Init/Jax Methods
    def __init__(self, pdf: ArrayLike, rho: ArrayLike, vel: ArrayLike, pdf_eq: ArrayLike, faces_index: ArrayLike):
        super().__init__(pdf,rho,vel)

        # Defining Cell Vars.
        self.pdf_eq = pdf_eq

        # STATIC VARIABLES #
        #define faces in cell # should be an ArrayLike['int']
        self.faces_index = faces_index

    @classmethod
    def pdf_init(cls,pdf:ArrayLike,faces_index:ArrayLike):
        temp = cls.__new__(cls)

        temp.pdf = pdf
        temp.rho = temp.density(pdf)
        temp.vel = temp.velocity(pdf,temp.rho)

        temp.pdf_eq = temp.equilibrium(temp.rho,temp.vel)
        temp.faces_index = faces_index
        return temp
        
    def super_init(self,pdf,rho,vel):
        super().__init__(pdf,rho,vel)

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        children += (self.pdf_eq,)
        aux_data.update({'faces_index': self.faces_index})
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    ### END ###
    
    # Class Methods
    def get_neq_pdf(self): # returns non-equilibrium pdf
        return self.pdf-self.pdf_eq
    
    def calc_eq(self):
        self.pdf_eq = self.equilibrium(self.rho,self.vel)

    def calc_eq_from_flux(self,faces:ArrayLike): # calculates eq using face flux
        pass

    def update_Cell(self):
        pass
    
class Face(Element):

    #Init/Jax Methods   
    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike,
                 flux:ArrayLike,nodes_index:ArrayLike,stencil_cells_index:ArrayLike,stencil_dists: ArrayLike):
        super().__init__(pdf,rho,vel)
        # Defining Face Vars.

        self.flux = flux

        # STATIC VARIABLES #
        self.nodes_index = nodes_index
        self.stencil_cells_index = stencil_cells_index
        self.stencil_dists = stencil_dists

    @classmethod
    def pdf_init(cls,pdf:ArrayLike,nodes_index:ArrayLike,stencil_cells_index: ArrayLike,stencil_dists = ArrayLike):
        temp = cls.__new__(cls)

        temp.pdf = pdf
        temp.rho = temp.density(pdf)
        temp.vel = temp.velocity(pdf,temp.rho)

        temp.flux = jnp.zeros_like(pdf)
        temp.nodes_index = nodes_index
        temp.stencil_cells_index = stencil_cells_index
        temp.stencil_dists = stencil_dists
        return temp

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children,aux_data = super().tree_flatten()
        children += (self.flux,)
        aux_data.update({'nodes_index':self.nodes_index,'stencil_cells_index':self.stencil_cells_index,'stencil_dists': self.stencil_dists})
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, **aux_data)
    ### END ###
    
    # Class Methods
    def calc_eq_from_stencil(self): # via stenciling
        pass
    
    def calc_flux(self):
        pass

    def update_Face(self):
        pass

class Node(Element):

    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike,cells_index: ArrayLike, cell_dists: ArrayLike):
        super().__init__(pdf,rho,vel)

        # Defining Node Vars.
        self.cells_index = cells_index
        self.cell_dists = cell_dists

    @classmethod
    def pdf_init(cls,pdf:ArrayLike,cells_index:ArrayLike,cells_dists: ArrayLike):
        temp = cls.__new__(cls)

        temp.pdf = pdf
        temp.rho = temp.density(pdf)
        temp.vel = temp.velocity(pdf,temp.rho)

        temp.flux = jnp.zeros_like(pdf)
        temp.cells_index = cells_index
        temp.cell_dists = cells_dists
        return temp
    
    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        children += (self.pdf_eq,)
        aux_data.update({'cells_index': self.cells_index,'cell_dists': self.cell_dists})
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    ### END ###

    # Class Methods
    def calc_NonEq(self,cells):
        pass

    def set_boundary_macros(self): # should be implemented in subclass
        pass

    def calc_NodalPDF(self):
        return self.calc_NonEq()+self.equilibrium(self.rho,self.vel)

    def update_Node(self):
        self.set_boundary_macros()
        self.pdf = self.calc_NodalPDF