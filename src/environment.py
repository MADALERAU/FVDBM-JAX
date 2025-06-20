"""
environment
"""
import jax
import sys
sys.path.append(".")
from src.dynamics import D2Q9
from src.containers import *

@jax.tree_util.register_pytree_node_class
class Environment():
    # Static Class Var.
    dynamics: Dynamics

    def __init__(self, cells: Cells, faces: Faces, nodes: Nodes): # defines environment from list of cells, faces, and nodes
        self.cells = cells
        self.faces = faces
        self.nodes = nodes

    @classmethod
    def create(cls,num_cells,num_faces,num_nodes):
        ''' Factory method to create an Environment instance with specified number of cells, faces, and nodes. '''
        temp  = cls.__new__(cls)
        temp.cells = Cells(num_cells,cls.dynamics)
        temp.faces = Faces(num_faces,cls.dynamics)
        temp.nodes = Nodes(num_nodes,cls.dynamics)
        return temp
    
    def init(self):
        ''' Initialize the environment by initializing its cells, faces, and nodes. '''
        self.cells.init()
        self.faces.init()
        self.nodes.init()
    
    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children = (self.cells,self.faces,self.nodes)
        aux_data = {}
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    ### END ###
    
    @jax.jit
    def step(self):
        '''
        one iteration of Standard FVDBM
        '''
        self.cells.calc_macros() # calculate macroscopic variables from PDFs
        self.cells.calc_eqs()   # calculate equilibrium PDFs
        self.nodes.calc_pdfs(self.cells)    # calculate PDFs at nodes from cells
        self.faces.calc_fluxes(self.cells,self.nodes) # calculate fluxes at faces from cells and nodes
        self.cells.calc_pdfs(self.faces)    # calculate PDFs at cells from faces
        return self
    
    def __repr__(self):
        return f"Environment(cells={self.cells}, faces={self.faces}, nodes={self.nodes})"