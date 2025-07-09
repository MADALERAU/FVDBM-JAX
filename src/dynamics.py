"""
Boltzmann Dynamics
"""


import jax
from jax.typing import ArrayLike
from jax import Array
import jax.numpy as jnp
from functools import partial


class Dynamics():
    '''Base class for all Dynamics in the FVDBM solver.'''
    #Static
    DIM: int # Dimension of the lattice (2D or 3D)
    NUM_QUIVERS: int # Number of quivers in the lattice (e.g., 9 for D2Q9, 13 for D2Q13)
    KSI: Array   # Define in sub-class
    W: Array     # Define in sub-class
    C: float    # Speed of sound, defined in sub-class
    tau: Array # Relaxation time, defined in sub-class
    delta_t: Array # Time step, defined in sub-class

    def __init__(self):
        pass

    def calc_eq(self,rho: ArrayLike,vel:ArrayLike):
        '''Calculates the equilibrium distribution function based on density and velocity.'''
        pass

    def ones_pdf(self):
        '''Returns an array of ones with the size of NUM_QUIVERS.'''
        return jnp.ones(self.NUM_QUIVERS)
    
    def density(self,pdf: ArrayLike):
        '''Calculates the density from the PDF by taking sum of all quivers.'''
        return jnp.sum(pdf,keepdims=True)
    
    def velocity(self,pdf:ArrayLike,rho:ArrayLike):
        '''Calculates the velocity from the PDF and density.'''
        return jnp.dot(self.KSI.T,pdf)/rho

    def calc_macro(self,pdf: ArrayLike):
        '''Calculates and returns the density and velocity from the PDF.'''
        rho = self.density(pdf)
        vel = self.velocity(pdf,rho)
        return rho,vel

### D2Q9 Dynamics ###
class D2Q9(Dynamics):
    '''D2Q9 Dynamics for 2D Lattice Boltzmann Method.'''
    DIM = 2
    NUM_QUIVERS = 9
    KSI = jnp.array([[0,0],  # center
                     [1,0],  # right
                     [0,1],  # top
                     [-1,0], # left
                     [0,-1], # bottom
                     [1,1],  # top-right
                     [-1,1], # top-left
                     [-1,-1],# bot-left
                     [1,-1]])# bot-right
    W = jnp.array([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36])
    C = 1/jnp.sqrt(3)
    def __init__(self,tau,delta_t):
        super().__init__()
        self.tau = tau
        self.delta_t = delta_t
    
    def calc_eq(self,rho: ArrayLike,vel:ArrayLike):
        return self.W*rho*(1+
                           (jnp.dot(self.KSI,vel))/(self.C**2)+
                           (jnp.dot(self.KSI,vel))**2/(2*self.C**4)-
                           (jnp.dot(vel,vel))/(2*self.C**2))

    
class D2Q13(Dynamics):
    '''D2Q13 Dynamics for 2D Lattice Boltzmann Method.'''
    DIM = 2
    NUM_QUIVERS = 13
    KSI = jnp.array([[0,0],  # center
                     [1,0],  # right
                     [0,1],  # top
                     [-1,0], # left
                     [0,-1], # bottom
                     [1,1],  # top-right
                     [-1,1], # top-left
                     [-1,-1],# bot-left
                     [1,-1], # bot-right
                     [2,0],  # right-double size
                     [0,2],  # top-double size
                     [-2,0], # left-double size
                     [0,-2]])# bottom-double size
    W = jnp.array([3/8,1/12,1/12,1/12,1/12,1/16,1/16,1/16,1/16,1/96,1/96,1/96,1/96])
    C = 1/jnp.sqrt(2)
    def __init__(self,tau,delta_t):
        super().__init__()
        self.tau = tau
        self.delta_t = delta_t
    
    def calc_eq(self,rho: ArrayLike,vel:ArrayLike):
        return self.W*rho*(1+(jnp.dot(self.KSI,vel))/(self.C**2)+(jnp.dot(self.KSI,vel))**2/(2*self.C**4)-(jnp.dot(vel,vel))/(2*self.C**2)+(jnp.dot(self.KSI,vel))**3/(2*self.C**6)-3*(jnp.dot(self.KSI,vel))*(jnp.dot(vel,vel))/(2*self.C**4))
    

