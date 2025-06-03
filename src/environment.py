"""
environment
"""
import jax
import sys
sys.path.append(".")
from src.elements import *
from src.dynamics import D2Q9
from src.methods import *
from src.containers import *

class Environment(MultiElementContainer):
    # Static Class Var.
    dynamics: Dynamics

    def __init__(self, cells: Container, faces: Container, nodes: Container,  methods:Methods): # defines environment from list of cells, faces, and nodes
        self.cells = cells
        self.faces = faces
        self.nodes = nodes
        self.methods = methods

    @classmethod
    def create(cls,cells=[],faces = [],nodes = []):
        temp  = cls.__new__(cls)
        temp.cells = ElementContainer("cells",cells)
        temp.faces = ElementContainer("faces",faces)
        temp.nodes = NodeContainer("nodes",nodes)
        return temp
    
    def init(self):
        dynamic = {}
        static = {}
        dynamic["cells"],static["cells"] = self.cells.flatten()
        dynamic["faces"],static["faces"] = self.faces.flatten()
        dynamic["nodes"],static["nodes"] = self.nodes.flatten()
        return  dynamic,static
    
    def setMethods(self,method):
        self.methods = method(self.dynamics)

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        children = (self.cells,self.faces,self.nodes)
        aux_data = {}
        return (children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        return cls(*children,**aux_data)
    ### END ###
    
    @partial(jax.jit,static_argnums=(0))
    def step(self,params,config): # one iteration of FVDBM
        params = self.methods.calc_cell_macros(params,config)
        params = self.methods.calc_cell_eqs(params,config)
        params = self.methods.calc_node_pdfs(params,config)
        params = self.methods.calc_face_pdfs(params,config)
        params = self.methods.calc_cell_pdfs(params,config)
        return params