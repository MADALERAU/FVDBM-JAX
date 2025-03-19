"""
methods
"""

from src.dynamics import *
from utils.utils import *

class Methods:
    def __init__(self,dynamics:Dynamics):
        self.dynamics = dynamics

### Cell Methods
    def vmap_cells(self,func,params):
        return jax.vmap(func,in_axes=({"cells":0,"faces":None,"nodes":None},))(params)

    # Equilibrium PDF @ Cell Centroids
    def calc_cell_eq(self,params):
        params["cells"]["pdf_eq"] = self.dynamics.calc_eq(params["cells"]["rho"],params["cells"]["vel"])
        return params
    
    # Macro Velocity and Density @ Cell Centroid
    def calc_cell_macro(self,params):
        params["cells"]["rho"],params["cells"]["vel"] = self.dynamics.calc_macro(params["cells"]["pdf"])
        return params
    
    def calc_cell_eqs(self,params):
        return self.vmap_cells(self.calc_cell_eq,params)
    
    def calc_cell_macros(self,params):
        return self.vmap_cells(self.calc_cell_eq,params)

### Node Methods
    def vmap_nodes(self,func,params,config,type: str):
        params["nodes"][type] = jax.vmap(func,in_axes=(
            0,0,
            {"cells":None,"faces":None,"nodes":None},
            {"cells":None,"faces":None,"nodes":None}))(params["nodes"][type],config["nodes"][type],params,config)
        return params

    # Non-Equilibrium PDF @ Node
    def calc_node_neq(self,params,config):
        neq = self.get_cell_neqs(params,config["nodes"]["cells_index"])
        cell_dists = config["nodes"]["cell_dists"]
        return extrapolate(neq,cell_dists)
    
    def calc_vel_node(self,node_params,node_config,params,config):
        params["nodes"] = node_params
        config["nodes"] = node_config
        cell_dists = config["nodes"]["cell_dists"]
        rho = extrapolate(self.get_cell_vels(params,config["nodes"]["cells_index"]),cell_dists)
        vel = params["nodes"]["vel"]
        params["nodes"]["pdf"] = self.dynamics.calc_eq(rho,vel) + self.calc_node_neq(params,config)
        return params["nodes"]
    
### Master node function
    def calc_node_pdfs(self,params,config):
        params = self.vmap_nodes(self.calc_vel_node,params,config,"Velocity")
        return params

    def get_cell_pdf(self,params,index):
        return params["cells"]["pdf"][index]
    
    def get_cell_eq(self,params,index):
        return params["cells"]["pdf_eq"][index]
    
    def get_cell_neq(self,params,index):
        return jax.lax.select(index==-1,jnp.zeros((params["cells"]["pdf_eq"].shape[-1])),self.get_cell_pdf(params,index)-self.get_cell_eq(params,index))
    
    def get_cell_vel(self,params,index):
        return jax.lax.select(index==-1,jnp.zeros((params["cells"]["vel"].shape[-1])),params["cells"]["vel"][index])

    def get_cell_rho(self,params,index):
        return jax.lax.select(index==-1,jnp.zeros((params["cells"]["rho"].shape[-1])),params["cells"]["rho"][index])

    def get_cell_neqs(self,params,indices):
        return jax.vmap(self.get_cell_neq,in_axes=(None,0))(params,indices)
    
    def get_cell_vels(self,params,indices):
        return jax.vmap(self.get_cell_vel,in_axes=(None,0))(params,indices)
    
    def get_class_rhos(self,params,indices):
        return jax.vmap(self.get_cell_rho,in_axes=(None,0))(params,indices)