import jax
from src.fvdbmElements import *

class Environment():
    def __init__(self,cells = [],faces = [],nodes = []): # defines environment from list of cells, faces, and nodes
        self.cells = cells
        self.faces = faces
        self.nodes = nodes

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children = (self.cells,self.faces,self.nodes)
        aux_data = {}
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    ### END ###
    
    def step(self): # one iteration of FVDBM

        # Temp: Later update with parallel loops
        for cell in self.cells:
            cell.calc_macro(self)
            cell.calc_eq(self)
        
        for node in self.nodes:
            node.update_Node()
        
        for face in self.faces:
            face.update_Face()
        
        for cell in self.cells:
            cell.update_Cell()