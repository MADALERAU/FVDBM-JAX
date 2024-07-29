import jax
from src.fvdbmElements import *

class Environment():
    def __init__(self,cells = [],faces = [],nodes = []):
        self.cells = cells
        self.faces = faces
        self.nodes = nodes

    def tree_flatten(self):
        children = (self.cells,self.faces,self.nodes)
        aux_data = {}
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)

    @jax.jit
    def calc_cell_eqs(self):
        return jax.pmap(lambda x: x.equilibrium)(self.cells)
    