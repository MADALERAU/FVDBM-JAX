"""
methods
"""

from src.dynamics import *
from utils.utils import *
from src.containers import *

from jax.tree_util import register_pytree_node_class

@register_pytree_node_class
class Methods:
    def __init__(self,dynamics:Dynamics):
        self.dynamics = dynamics

    ### JAX PyTree Methods ###
    def tree_flatten(self):
        """
        @private JAX Pytree Handling Method
        """
        children = ()
        aux_data = {self.dynamics}
        return(children,aux_data)
    
    @classmethod
    def tree_unflatten(cls,aux_data,children):
        """
        @private JAX Pytree Handling Method
        """
        return cls(*children,**aux_data)
    ### END ###

### Cell Methods
    def vmap_cells(self,func,params,config):
        return jax.vmap(func,in_axes=({"cells":0,"faces":None,"nodes":None},{"cells":0,"faces":None,"nodes":None}))(params,config)

    # Equilibrium PDF @ Cell Centroids
    def calc_cell_eq(self,params,config):
        return self.dynamics.calc_eq(params["cells"]["rho"],params["cells"]["vel"])
    
    # Macro Velocity and Density @ Cell Centroid
    def calc_cell_macro(self,params,config):
        return self.dynamics.calc_macro(params["cells"]["pdf"])
    
    def calc_cell_eqs(self,params,config):
        params["cells"]["pdf_eq"] = self.vmap_cells(self.calc_cell_eq,params,config)
        return params
    
    def calc_cell_macros(self,params,config):
        params["cells"]["rho"],params["cells"]["vel"] = self.vmap_cells(self.calc_cell_macro,params,config)
        return params

    def calc_cell_pdf(self,params,config):
        faces_ind = config["cells"]["faces_index"]
        faces_n = config["cells"]["faces_n"]
        fluxes = jax.vmap(self.get_face_flux,in_axes=(None,0,0))(params,faces_ind,faces_n)
        #print(fluxes)
        flux = jnp.sum(fluxes,axis=0)
        return params["cells"]["pdf"]+self.dynamics.delta_t*(1/self.dynamics.tau*(params["cells"]["pdf_eq"]-params["cells"]["pdf"])-flux)
    
    def calc_cell_pdfs(self,params,config):
        params["cells"]["pdf"] = self.vmap_cells(self.calc_cell_pdf,params,config)
        return params
    
    def get_face_flux(self,params,face_ind,face_n):
        return params["faces"]["flux"][face_ind]*face_n

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
        rho = extrapolate(self.get_cell_rhos(params,config["nodes"]["cells_index"]),cell_dists)
        vel = params["nodes"]["vel"]
        params["nodes"]["pdf"] = self.dynamics.calc_eq(rho,vel) + self.calc_node_neq(params,config)
        params["nodes"]["rho"] = rho
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
        return jax.lax.select(index==-1,jnp.asarray([0.]),params["cells"]["rho"][index])

    def get_cell_neqs(self,params,indices):
        return jax.vmap(self.get_cell_neq,in_axes=(None,0))(params,indices)
    
    def get_cell_vels(self,params,indices):
        return jax.vmap(self.get_cell_vel,in_axes=(None,0))(params,indices)
    
    def get_cell_rhos(self,params,indices):
        return jax.vmap(self.get_cell_rho,in_axes=(None,0))(params,indices)
    
## Face Methods
    def vmap_faces(self,func,params,config):
        return jax.vmap(func,in_axes=({"cells":None,"faces":0,"nodes":None},
                                      {"cells":None,"faces":0,"nodes":None}))(params,config)
    
    def test(self,params,config):
        return params
    
    def test_vmap(self,params,config):
        return self.vmap_faces(self.test,params,config)
    
    def calc_face_pdf(self,params,config):
        cells_index = config["faces"]["stencil_cells_index"]
        cells_dist = config["faces"]["stencil_dists"]

        left_cell_pdf = jax.lax.select(cells_index[0]==-1,self.calc_ghost(params,config,cells_index[1],cells_dist[1],cells_dist[0]),self.get_cell_pdf(params,cells_index[0]))
        right_cell_pdf = jax.lax.select(cells_index[1]==-1,self.calc_ghost(params,config,cells_index[0],cells_dist[0],cells_dist[1]),self.get_cell_pdf(params,cells_index[1]))
        pdf = jnp.zeros_like(left_cell_pdf)

        norm = jnp.dot(self.dynamics.KSI,config["faces"]["n"])
        interp_cell = interp_pdf(jnp.stack([left_cell_pdf,right_cell_pdf],axis=0),cells_dist)

        pdf = jnp.where(norm>0,left_cell_pdf,right_cell_pdf)

        pdf = jnp.where(norm==0,interp_cell,pdf)

        flux = pdf*norm*config["faces"]["L"]
        return pdf,flux

    def calc_face_pdfs(self,params,config):
        params["faces"]["pdf"],params["faces"]["flux"] = self.vmap_faces(self.calc_face_pdf,params,config)
        return params

    def calc_ghost(self,params,config,known_cell_ind,known_cell_dist,ghost_cell_dist):
        nodes_index = config["faces"]["nodes_index"]
        face_pdf = jnp.mean(self.get_node_pdfs(params,nodes_index),axis=0)
        cell_pdf = self.get_cell_pdf(params,known_cell_ind)
        return extrap_pdf(face_pdf,cell_pdf,ghost_cell_dist,known_cell_dist)


    def get_node_pdf(self,params,index):
        def func_internal(ind):
            return params["nodes"]["Internal"]["pdf"][ind]
        def func_velocity(ind):
            return params["nodes"]["Velocity"]["pdf"][ind]
        def func_pressure(ind):
            return params["nodes"]["Pressure"]["pdf"][ind]
        
        func_list = [func_internal,
                     func_velocity,
                     func_pressure]
        
        return jax.lax.switch(index[0],func_list,index[1])
    
    def get_node_pdfs(self,params,indices):
        return jax.vmap(self.get_node_pdf,in_axes=(None,0))(params,indices)