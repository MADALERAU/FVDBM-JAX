import jax.numpy as jnp
from src.fvdbmDynamics import Dynamics
from jax.typing import ArrayLike
from jax import Array

class Element():

    # Static
    dynamics: Dynamics

    def __init__(self):
        self.pdf = self.dynamics.ones_pdf()
        self.rho = self.density()
        self.vel = self.velocity()

    def __str__(self):
        return f'Element(Density: {self.rho}, Velocity: {self.vel.squeeze()})'

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


class Cell(Element):

    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike):
        super().__init__()
        # Defining Parent(Element) Vars.
        self.pdf = pdf
        self.rho = rho
        self.vel = vel

        # Defining Cell Vars.
        self.pdf_eq = self.equilibrium()

    def get_neq_pdf(self): # returns non-equilibrium pdf
        return self.pdf-self.pdf_eq
    
class Face(Element):

    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike):
        super().__init__()
        # Defining Parent(Element) Vars.
        self.pdf = pdf
        self.rho = rho
        self.vel = vel

        # Defining Face Vars.
        self.nodes: Array['Nodes'] = []
        self.stencil_cells: Array[Cell]=[]
        self.stencil_dist: Array[float]=[]

class Node(Element):

    def __init__(self,pdf:ArrayLike,rho:ArrayLike,vel:ArrayLike):
        super().__init__()
        self.pdf = pdf
        self.rho = rho
        self.vel = vel

        # Defining Node Vars.
        self.cells: Array[Cell] = []
        self.cell_dists: Array[float] = []

    