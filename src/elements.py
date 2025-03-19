"""
elements
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

class Element():
    """
    Generic class for all Cells, Nodes, Faces, Etc
    """
    # Static
    dynamics: Dynamics
    """
    Defined by the `FVDBMSolver.src.dynamics` classes. Sets preference for collision and lattice to predefine relevant array sizes and methods.
    """

    # Init/Jax Methods
    def __init__(self,pdf,rho,vel):
        """
        Base initialization classes for flexible JAX Pytree handling.
        ___
        **Notes:** The __init__ function should not be used in implementation.
        """
        self.pdf = pdf
        """
        @private
        """
        self.rho = rho
        """
        @private
        """
        self.vel = vel
        """
        @private
        """

    @classmethod
    def pdf_init(cls,pdf:jax.Array):
        """
        Creates an Element instance using a given PDF. Defines element density and velocity from input PDF.
        ___
        **Parameters:**
        - **pdf** (ArrayLike): Input PDF.

        **Returns:** A new Element.
        ___
        """
        temp = cls.__new__(cls)
        temp.pdf = pdf
        # temp.rho = None
        # temp.vel = None
        temp.calc_macro()
        return temp

    @classmethod
    def ones_init(cls): # for init w/ pdf as ones
        """
        Initializes an Element instance with a PDF of ones. Defines element density and velocity from ones PDF.
        ___
        **Returns:** A new Element.
        ___
        """
        temp = cls.__new__(cls)
        temp.pdf = cls.dynamics.ones_pdf()
        # temp.rho = None
        # temp.vel = None
        temp.calc_macro()
        return temp
    
    ### Custom PyTree Methods ###
    def flatten(self):
        """
        Flattens the class element into a dynamic and static dictionary
        ___
        **Returns:**
        - **dynamic** (Dict): dynamic variables of element in Dictionary form.
        - **static** (Dict): static variables of element in Dictionary form.
        ___
        """
        dynamic = {"pdf": self.pdf,"rho": self.rho,"vel":self.vel}
        static = {}
        return dynamic, static

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        """
        @private JAX Pytree Handling Method
        """
        children = (self.pdf,self.rho,self.vel)
        aux_data = {}
        return(children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        """
        @private JAX Pytree Handling Method
        """
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

    ### Custom PyTree Methods ###
    def flatten(self):
        dynamic, static = super().flatten()
        dynamic.update({"pdf_eq":self.pdf_eq})
        static.update({"faces_index":self.faces_index})
        return dynamic, static

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

    ### Custom PyTree Methods ###
    def flatten(self):
        dynamic, static = super().flatten()
        dynamic.update({"flux":self.flux})
        static.update({"nodes_index":self.nodes_index,"stencil_cells_index":self.stencil_cells_index,"stencil_dists":self.stencil_cells_index})
        return dynamic, static

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

class Node(Element):

    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike,type:ArrayLike, cells_index: ArrayLike, cell_dists: ArrayLike):
        super().__init__(pdf,rho,vel)

        # Defining Node Vars.
        self.type = type 
        self.cells_index = cells_index
        self.cell_dists = cell_dists

    @classmethod
    def pdf_init(cls,pdf:ArrayLike,type:int,cells_index:ArrayLike,cells_dists: ArrayLike):
        temp = cls.__new__(cls)

        temp.pdf = pdf
        temp.rho = temp.density(pdf)
        temp.vel = temp.velocity(pdf,temp.rho)

        temp.type = type
        temp.cells_index = cells_index
        temp.cell_dists = cells_dists
        return temp
    
    ### Custom PyTree Methods ###
    def flatten(self):
        dynamic, static = super().flatten()
        static.update({"type":self.type,"cells_index":self.cells_index,"cell_dists":self.cell_dists})
        return dynamic, static
    
    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        children += (self.pdf_eq,)
        aux_data.update({'type':self.type,'cells_index': self.cells_index,'cell_dists': self.cell_dists})
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children,**aux_data)
    ### END ###

    # Class Methods
