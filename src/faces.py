import jax
import jax.numpy as jnp
import sys
sys.path.append(".")
from src.dynamics import Dynamics
from utils.utils import *
from jax.tree_util import register_pytree_node_class
from src.containers import *

@register_pytree_node_class
class CCStencilFaces(Faces):
    """
    Class for faces with the new stencl scheme which uses adjacent cell centers as intersection points for the stencil.

    self.n represents the direction of the stencil, not the face normal.
    self.alpha represents the angle between n and n_PQ

    ### Notes: ###
    Currently Inoperable
    """
    def __init__(self,size,dynamics: Dynamics,flux_scheme = 'upwind'):
        super().__init__(size, dynamics,flux_scheme=flux_scheme)
        self.alpha = jnp.zeros((size,1),dtype=jnp.float32)  # Angle between n and n_PQs

    def init(self):
        super().init()
        self.alpha = jnp.asarray(self.alpha)
    
    # Jax Flattening Methods
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data['alpha'] = self.alpha
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = super().tree_unflatten(aux_data, children)

        obj.alpha = aux_data['alpha']
        return obj
    
    def calc_fluxes(self, cells, nodes):
        self.pdf = jax.vmap(self.calc_flux,in_axes=(None,None,0,0,0,0,0,0))(
            cells,
            nodes,
            self.stencil_cells_index,
            self.stencil_dists,
            self.nodes_index,
            self.n,
            self.L,
            self.alpha
        )

    def calc_flux(self,
                  cells: Cells,
                  nodes: Nodes,
                  cells_index,
                  cell_dists,
                  nodes_index,
                  n,
                  L,
                  alpha):
        match self.flux_scheme:
            case 'upwind':
                calc_flux = self.calc_upwind_flux
            case 'lax_wendroff':
                calc_flux = self.calc_lax_wendroff_flux
            case _:
                raise ValueError(f"Unknown flux scheme: {self.flux_scheme}")
        
        flux = calc_flux(cells,nodes,cells_index,cell_dists,nodes_index,n,L)
        flux = flux*jnp.cos(alpha)

        return flux

    ### Implementation Notes: Calc Ghost doesn't need to be overridden if all edge faces have centered pdfs for stencil definition.

@register_pytree_node_class
class CCStencilKsiFaces(Faces):
    """
    Alternative class for faces with the new stencil scheme which uses adjacent cell centers as intersection points for the stencil.
    
    ### Notes: ###
    Currently Inoperable
    """
    def __init__(self,size,dynamics: Dynamics,flux_scheme = 'upwind'):
        super().__init__(size, dynamics,flux_scheme=flux_scheme)
        self.npq = jnp.zeros((size,dynamics.DIM),dtype=jnp.float32)  # Angle between n and n_PQs

    def init(self):
        super().init()
        self.npq = jnp.asarray(self.npq)

    # Jax Flattening Methods
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()
        aux_data['npq'] = self.npq
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = super().tree_unflatten(aux_data, children)

        obj.npq = aux_data['npq']
        return obj
    
    def calc_fluxes(self, cells, nodes):
        self.pdf = jax.vmap(self.calc_flux,in_axes=(None,None,0,0,0,0,0,0))(
            cells,
            nodes,
            self.stencil_cells_index,
            self.stencil_dists,
            self.nodes_index,
            self.n,
            self.L,
            self.npq
        )

    def calc_flux(self,
                  cells: Cells,
                  nodes: Nodes,
                  cells_index,
                  cell_dists,
                  nodes_index,
                  n,
                  L,
                  npq):
        match self.flux_scheme:
            case 'upwind':
                calc_flux = self.calc_upwind_flux
            case 'lax_wendroff':
                calc_flux = self.calc_lax_wendroff_flux
            case _:
                raise ValueError(f"Unknown flux scheme: {self.flux_scheme}")
        
        flux = calc_flux(cells,nodes,cells_index,cell_dists,nodes_index,n,L)
        alpha = jnp.concatenate([jnp.zeros_like(n[...,0:1]),jnp.dot(self.dynamics.KSI[1:],n)/jnp.dot(self.dynamics.KSI[1:],npq)])
        
        flux = flux*alpha

        return flux