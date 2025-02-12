import jax
import sys
sys.path.append(".")
from src.elements import *
from src.dynamics import D2Q9

class Environment():
    # Static
    dynamics: Dynamics

    def __init__(self,cells: ElementContainer,faces: ElementContainer,nodes: ElementContainer): # defines environment from list of cells, faces, and nodes
        self.cells = cells
        self.faces = faces
        self.nodes = nodes

        

    @classmethod
    def create(cls,cells=[],faces = [],nodes = []):
        temp = cls(ElementContainer(cells),
                   ElementContainer(faces),
                   ElementContainer(nodes))
        return temp
    
    def init(self):
        return {"cells": self.cells.flatten(),"faces": self.cells.flatten(),"nodes":self.nodes.flatten()}

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

    # Cell Methods
    @classmethod
    def vmap_cells(cls,func,params):
        return jax.vmap(func,in_axes=({"cells":0,"faces":None,"nodes":None},))(params)

    @classmethod
    def calc_cell_eq(cls,params):
        params["cells"]["pdf"] = cls.dynamics.calc_eq(params["cells"]["rho"],params["cells"]["vel"])
        return params
    
    @classmethod
    def calc_cell_eqs(cls,params):
        return cls.vmap_cells(cls.calc_cell_eq,params)
    

    # Node Methods
    @classmethod
    def vmap_nodes(cls,func,params):
        return jax.vmap(func,in_axes=({"cells":None,"face":None,"nodes":0},))(params)
    
    @classmethod
    def calc_nodes_neq(cls,params):
        pdfs = cls.get_cell_pdfs(params,params["nodes"]["cells_index"])
        cell_dists = params["nodes"]["cells_dists"]

    @classmethod
    def get_cell_pdf(cls,params,index):
        return jax.lax.cond(index==-1,jnp.zeros((params["cells"]["pdf"].shape[-1])),params["cells"]["pdf"][index])

    @classmethod
    def get_cell_pdfs(cls,params,indices):
        return jax.vmap(cls.get_cell_pdf,in_axes=(None,0))(params,indices)