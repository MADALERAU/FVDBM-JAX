"""
containers...
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike
from jax import Array, jit
from functools import partial
import sys
sys.path.append(".")
from src.dynamics import Dynamics
from utils.utils import *
import jax.tree_util
import numpy as np

@jax.tree_util.register_pytree_node_class
class Container():
    dynamics: Dynamics
    '''
    Generic class for all Containers, such as Cells, Nodes, Faces, etc.
    '''
    def __init__(self,size,dynamics: Dynamics):
        '''
        Initializes the Container with a given size and dynamics
        '''
        self.pdf = jnp.ones((size, dynamics.NUM_QUIVERS), dtype=jnp.float32)
        self.dynamics = dynamics

    # JAX flattening and unflattening functions for Container
    def tree_flatten(self):
        children = (self.pdf,)
        aux_data = {'dynamics': self.dynamics}
        return children, aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.pdf = children[0]
        obj.dynamics = aux_data['dynamics']
        return obj
    
    def __repr__(self):
        return repr(self.__dict__)
    
@jax.tree_util.register_pytree_node_class
class Cells(Container):
    '''
    Class for Cells in the FVDBM solver.
    Inherits from Container and initializes with the dynamics.
    '''
    def __init__(self, size, dynamics: Dynamics):
        super().__init__(size, dynamics)
        # Density and Velocity
        self.rho = jnp.zeros((size, 1), dtype=jnp.float32)
        self.vel = jnp.zeros((size, dynamics.DIM), dtype=jnp.float32)
        self.pdf_eq = jnp.zeros((size, dynamics.NUM_QUIVERS), dtype=jnp.float32)

        self.face_indices = CustomArray(size, dtype=jnp.int32, default_value=-1)

        self.face_normals = CustomArray(size, dtype=jnp.int32, default_value=-1)

    def init(self):
        '''
        Initializes Cell container for the FVDBM solver.
        '''
        self.face_indices = jnp.asarray(self.face_indices)
        self.face_normals = jnp.asarray(self.face_normals)

    # Jax flattening and unflattening functions for Cells
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()

        children += (self.rho, self.vel,self.pdf_eq)

        aux_data["face_indices"] = self.face_indices
        aux_data["face_normals"] = self.face_normals
        return children, aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = super().tree_unflatten({"dynamics":aux_data["dynamics"]}, children[:1])

        obj.rho = children[1]
        obj.vel = children[2]
        obj.pdf_eq = children[3]
        obj.face_indices = aux_data["face_indices"]
        obj.face_normals = aux_data["face_normals"]

        return obj
    
    # Cell Methods
    def calc_macros(self):
        '''
        Calculates the macroscopic properties (density and velocity) of the cells.
        Uses JAX's vmap for vectorized computation.
        '''
        self.rho,self.vel = jax.vmap(self.dynamics.calc_macro)(self.pdf)

    def calc_eqs(self):
        '''
        Calculates the equilibrium PDFs for the cells.
        Uses JAX's vmap for vectorized computation.
        '''
        self.pdf_eq = jax.vmap(self.dynamics.calc_eq)(self.rho, self.vel)

    def calc_pdfs(self,faces:Faces):
        '''
        vmapped wrapper for the calc_pdf method.
        Calculates the PDFs for all cells based on the equilibrium PDFs and fluxes from the faces.
        Uses JAX's vmap for vectorized computation.
        '''
        self.pdf = jax.vmap(self.calc_pdf,in_axes=(None,0,0,0,0))(faces,self.pdf,self.pdf_eq,self.face_indices, self.face_normals)
    
    def calc_pdf(self,faces:Faces, pdf, pdf_eq, face_indices, face_normals):
        '''
        Calculates the PDF for a cell based on the equilibrium PDF and fluxes from the faces.
        '''
        fluxes = jax.vmap(faces.get_flux)(face_indices)*face_normals[...,jnp.newaxis]
        flux = jnp.sum(fluxes, axis=0)
        return pdf + self.dynamics.delta_t * (1 / self.dynamics.tau * (pdf_eq - pdf) - flux)

    #getters
    def get_rho(self,cell_index):
        return self.rho[cell_index]
    
    def get_vel(self,cell_index):
        return self.vel[cell_index]
    
    def get_pdf(self,cell_index):
        return self.pdf[cell_index]
    
    def get_eq(self,cell_index):
        return self.pdf_eq[cell_index]

@jax.tree_util.register_pytree_node_class
class Faces(Container):
    '''
    Class for Faces in the FVDBM solver.
    Inherits from Container and initializes with the dynamics.

    Utilizes a first order upwind/downwind scheme for flux calculations.

    *** Faces.pdf is treated as flux in the FVDBM solver. ***
    '''
    def __init__(self, size, dynamics: Dynamics,flux_scheme: str = "upwind"):
        super().__init__(size, dynamics) # treats PDF as flux

        #Static
        self.nodes_index = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.stencil_cells_index = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.stencil_dists = CustomArray(size, dtype=jnp.float32, default_value=-1)
        self.n = jnp.zeros((size, dynamics.DIM), dtype=jnp.float32)
        self.L = jnp.zeros((size, 1), dtype=jnp.float32)
        self.flux_scheme = flux_scheme

    def init(self):
        '''
        Initializes Face container for the FVDBM solver.
        '''
        self.nodes_index = jnp.asarray(self.nodes_index)
        self.stencil_cells_index = jnp.asarray(self.stencil_cells_index)
        self.stencil_dists = jnp.asarray(self.stencil_dists)

    # Jax Flattening Methods
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()

        aux_data["nodes_index"] = self.nodes_index
        aux_data["stencil_cells_index"] = self.stencil_cells_index
        aux_data["stencil_dists"] = self.stencil_dists
        aux_data["n"] = self.n
        aux_data["L"] = self.L
        aux_data["flux_scheme"] = self.flux_scheme

        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = super().tree_unflatten({"dynamics":aux_data["dynamics"]}, children[:1])

        obj.nodes_index = aux_data["nodes_index"]
        obj.stencil_cells_index = aux_data["stencil_cells_index"]
        obj.stencil_dists = aux_data["stencil_dists"]
        obj.n = aux_data["n"]
        obj.L = aux_data["L"]
        obj.flux_scheme = aux_data["flux_scheme"]

        return obj
    
    def calc_fluxes(self,cells:Cells,nodes:Nodes):
        '''
        vmapped wrapper for the calc_flux method.
        Calculates the fluxes for all faces based on the cells and nodes.
        Uses JAX's vmap for vectorized computation.
        '''
        match self.flux_scheme:
            case "upwind":
                calc_flux = self.calc_upwind_flux
            case "lax_wendroff":
                calc_flux = self.calc_lax_wendroff_flux
            case _:
                raise ValueError(f"Unknown flux scheme: {self.flux_scheme}")
        
        self.pdf = jax.vmap(calc_flux,in_axes=(None,None,0,0,0,0,0))(
            cells,
            nodes,
            self.stencil_cells_index,
            self.stencil_dists,
            self.nodes_index,
            self.n,
            self.L
        )
    
    def calc_upwind_flux(self,
                       cells:Cells,
                       nodes:Nodes,
                       cells_index,
                       cell_dists,
                       nodes_index,
                       n,
                       L):
        '''
        Calculates the flux for a face based on the cells and nodes. Uses a first order extrapolation method
        left is upwind, right is downwind to the normal
        '''
        left_cell_pdf = jax.lax.select(cells_index[0]==-1,
                                       self.calc_ghost(cells,nodes,nodes_index,cells_index[1],cell_dists[0],cell_dists[1]),
                                       cells.get_pdf(cells_index[0]))
        right_cell_pdf = jax.lax.select(cells_index[1]==-1,
                                        self.calc_ghost(cells,nodes,nodes_index,cells_index[0],cell_dists[1],cell_dists[0]),
                                        cells.get_pdf(cells_index[1]))

        pdf = jnp.zeros_like(left_cell_pdf)
        norm = jnp.dot(self.dynamics.KSI, n)
        pdf = jnp.where(norm >= 0, left_cell_pdf, right_cell_pdf)

        flux = pdf*norm*L

        return flux
    
    def calc_lax_wendroff_flux(self,
                       cells:Cells,
                       nodes:Nodes,
                       cells_index,
                       cell_dists,
                       nodes_index,
                       n,
                       L):
        '''
        Calculates the flux for a face using the general Lax-Wendroff scheme (see screenshot equation).
        '''
        # Upwind and Downwind cell indices
        upwind_idx = cells_index[0]
        downwind_idx = cells_index[1]

        # Handle ghost cells
        pdf_U = jax.lax.select(upwind_idx == -1,
                               self.calc_ghost(cells, nodes, nodes_index, downwind_idx, cell_dists[0], cell_dists[1]),
                               cells.get_pdf(upwind_idx))
        pdf_D = jax.lax.select(downwind_idx == -1,
                               self.calc_ghost(cells, nodes, nodes_index, upwind_idx, cell_dists[1], cell_dists[0]),
                               cells.get_pdf(downwind_idx))

        # Distances
        x_C_minus_x_U = cell_dists[0]  # upwind cell center to face center (positive)
        x_D_minus_x_U = cell_dists[0] + cell_dists[1]  # upwind to downwind cell center (positive)

        # Characteristic speed and time step
        varpi = jnp.dot(self.dynamics.KSI, n)
        delta_t = self.dynamics.delta_t

        # Lax-Wendroff interpolation using explicit upwind/downwind ordering and positive distances
        interp = (x_C_minus_x_U / x_D_minus_x_U) - (varpi * delta_t) / (2 * x_D_minus_x_U)
        pdf_C = pdf_U + (pdf_D - pdf_U) * interp
        # Flux
        flux = pdf_C * varpi * L
        return flux

    def calc_ghost(self,cells:Cells,nodes: Nodes,nodes_index,known_cell_ind,ghost_cell_dist,known_cell_dist):
        '''
        Calculates the ghost cell PDF based on the known cell PDF and the distance to the ghost cell.
        Uses a first order extrapolation method.
        '''
        face_pdf = jnp.mean(jax.vmap(nodes.get_pdf)(nodes_index),axis=0)
        cell_pdf = cells.get_pdf(known_cell_ind)
        return extrap_pdf(face_pdf,cell_pdf,ghost_cell_dist,known_cell_dist)

    #getters
    def get_flux(self,face_index):
        return self.pdf[face_index]

@jax.tree_util.register_pytree_node_class
class Nodes(Container):
    '''
    Class for Nodes in the FVDBM solver.
    Inherits from Container and initializes with the dynamics.
    '''
    def __init__(self, size, dynamics: Dynamics):
        super().__init__(size, dynamics)
        # Density and Velocity
        self.rho = jnp.zeros((size, 1), dtype=jnp.float32)
        self.vel = jnp.zeros((size, dynamics.DIM), dtype=jnp.float32)

        self.type = jnp.zeros((size, 1), dtype=jnp.int32)  # Node type (e.g., boundary, internal)
        self.cells_index = CustomArray(size, dtype=jnp.int32, default_value=-1)
        self.cell_dists = CustomArray(size, dtype=jnp.float32, default_value=-1)

    def init(self):
        '''
        Initializes Node container for the FVDBM solver.
        '''
        self.cells_index = jnp.asarray(self.cells_index)
        self.cell_dists = jnp.asarray(self.cell_dists)
        #self.type = jax.nn.one_hot(self.type,jnp.unique(self.type).shape[0], dtype=jnp.int32)
        
    # Jax Flattening Methods
    def tree_flatten(self):
        children, aux_data = super().tree_flatten()

        children += (self.rho, self.vel)

        aux_data["type"] = self.type
        aux_data["cells_index"] = self.cells_index
        aux_data["cell_dists"] = self.cell_dists
        return children, aux_data
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = super().tree_unflatten({"dynamics":aux_data["dynamics"]}, children[:1])

        obj.rho = children[1]
        obj.vel = children[2]
        obj.type = aux_data["type"]
        obj.cells_index = aux_data["cells_index"]
        obj.cell_dists = aux_data["cell_dists"]

        return obj
    
    def calc_pdfs(self,cells:Cells):
        self.calc_vel_pdfs(cells)

    def calc_vel_pdfs(self,cells:Cells):
        indices = jnp.arange(self.pdf.shape[0]) # Assuming type 1 is for velocity nodes
        self.rho = jnp.where(self.type==1,jax.vmap(self.calc_density,in_axes=(None,0))(cells, indices), self.rho)

        self.pdf = jnp.where(self.type==1,jax.vmap(self.calc_vel_pdf,in_axes=(None,0))(cells, indices), self.pdf)

    def calc_vel_pdf(self,cells:Cells,index):
        rho = self.rho[index]
        vel = self.vel[index]
        pdf = self.dynamics.calc_eq(rho, vel)+self.calc_neq(cells,index)
        return pdf

    def calc_neq(self, cells: Cells, index):
        neqs = self.get_cell_neqs(self.cells_index[index], cells)
        return extrapolate(neqs, self.cell_dists[index])

        
    def calc_density(self,cells:Cells,index):
        '''
        Calculates the density of the node given its index.
        '''
        return extrapolate(self.get_cell_rhos(self.cells_index[index], cells),self.cell_dists[index])
    
    def get_cell_neqs(self, cell_indices, cells: Cells):
        '''
        Returns the non-equilibrium PDFs of the cells given their indices.
        Uses JAX's vmap for vectorized computation.
        '''
        return jax.vmap(self.get_cell_neq, in_axes=(0, None))(cell_indices, cells)

    def get_cell_neq(self,cell_index, cells: Cells):
        '''
        Returns the non-equilibrium PDFs of the cells given their indices.
        Uses JAX's vmap for vectorized computation.
        '''
        return jax.lax.select(cell_index==-1,
                              jnp.zeros((cells.pdf_eq.shape[-1])),
                              cells.get_pdf(cell_index)-cells.get_eq(cell_index))

    def get_cell_rhos(self,cell_indices, cells: Cells):
        '''
        Returns the densities of the cells given their indices.
        Uses JAX's vmap for vectorized computation.
        '''
        return jax.vmap(cells.get_rho)(cell_indices)
    
    def get_cell_vels(self,cell_index,cells: Cells):
        '''
        Returns the velocities of the cells given their indices.
        Uses JAX's vmap for vectorized computation.
        '''
        return jax.vmap(cells.get_vel)(cell_index)

    #getters
    def get_pdf(self,node_index):
        return self.pdf[node_index]