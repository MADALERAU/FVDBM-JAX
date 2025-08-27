# FVDBMSolver
Finite Volume Discrete Boltzmann Method Solver 

## Getting Started
The src module contains all core components of the FVDBM solver, with additional utility methods in the utils module. Please refer to [`ref/requirements.txt`](ref/requirements.txt) for all python dependencies.

The setup/solving process can be seperated into three sections: 
- mesh generation
- mesh initialization
- solver

Below is a quick description on each section.

### Mesh Generation
While the rest of the solver can utilize an external mesh given the point locations and cell/face connectivity info, the solver offers meshing and refining through [meshpy](https://documen.tician.de/meshpy/) which wraps [Triangle](https://www.cs.cmu.edu/~quake/triangle.html). A typical workflow example is given below:

```
from meshpy import triangle, geometry
from src.refiner import MeshRefiner

# Define meshpy geometry builder
builder = geometry.GeometryBuilder()

# Add geometries to the builder
builder.add_geometry(points,facets,facet_markers = facet_markers) # See .ipynb exmaples 

# Define meshpy MeshInfo
info = triangle.MeshInfo()
builder.set(info)

info.set_holes(np.asarray(holes)) # iterable list of all holes in domain (eg. [(0,0),(1,2),...])

# build mesh
mesh = triangle.build(info,generate_faces,...) # See examples for additional args used.
```
The resulting mesh object will contain the point and conectivity info needed for the solver. This info can be easily passed to the [`Mesher`](src/mesher.py) class by via

```
mesher = Mesher()
mesher.import_meshpy(mesh)
```

**Note: Often, Triangle struggles to generate the mesh without crashing. A few reruns of the file may be necessary to successfully generate a mesh.**

Alternatively, the point and connectivity info can be directly imported as follows:

```
mesher = Mesher()
mesher.points = points
mesher.cells = cell_point_indices
mesher.faces = face_point_indices

# Enforces ccw ordering
mesher.enforce_ccw()
```

#### Mesh Refining
For meshes generated using triangle, the [`MeshRefiner`](src/refiner.py) function offers a convenient way to refine and visualize the mesh quality. *Note that some functionality within the refiner may not behave as expected due to the LLM generated code* In general, the refining process is as follows:

```
# Create refiner object with meshpy generated mesh object
refiner = MeshRefiner(mesh, np.asarray(holes))

# Visualize Mesh Quality
refiner.show_mesh_quality()

# Mesh Improvement
mesh = improve(...) # See .ipynb files for example args.

# Visualize New Mesh Quality
refiner.show_mesh_quality()
```

### Mesh Initialization
After importing the final mesh to the [`Mesher`](src/mesher.py), the following code must be run to calculate all relevant geometry information.

```
mesher.calc_mesh_properties()
mesher.verify_stencil_geometry()
```
Then, relevant dynamics and parameters can be defined and passed into the solver. Parameters are typically passed in to [`Dynamics`](src/dynamics.py) and boundary conditions are passed through [`Mesher`](src.mesher.py) to [`Nodes`](src/nodes.py).

The general process is outlined below:

```
dynamics = D2Q9(tau=Tau,delta_t = dt) # Tau and dt defined seperately

cells,faces,nodes = mesher.to_env(dynamics,...) # additional args to define types of cells,faces,nodes

# Boundary Definition (see examples for more detail)
nodes = mesher.set_vel_node(nodes,marker=Marker,...)
nodes = mesher.set_rho_node(nodes,marker=Marker2,...)
```

### Solving
With all cells, faces, and nodes created, the environemnt can be defined and the simulation ran.

```
env = Environment(cells,faces,nodes)
env.init()

# loop for running sim
for i in range(iter):
    env = env.step()
```
The `env.step()` function returns itself as the output to comply with jax.jit functionality. In any intermediary steps, or prior to simulation, the mesher and env can be saved as a pickle file (mesher.to_pickle(env,'file_name')), the cell density and velocities saved as a vtk file(mesher.to_vtk(env,'file_name')), or the env variables visualized using `matplotlib`. For further info please refer to the examples in [`tests`](tests).

### etc
Please see the jupyter notebook (.ipynb) files under tests for examples on setting up and running cases.

**Note that some files may be outdated and inoperable. as of 8/27, the [`porous_flow.ipynb`](tests/porous_flow.ipynb) is the most recent example.**

## Contributing
This section highlights crucial aspects of the solver for future development as well as development notes and tips.

### Adding Solver Functionality
For most new functionality (eg. flux schemes, boundary conditions, multi-component flow), there are two important aspects to consider: dynamic and static variables. Dynamic variables refer to values which may change during simulation time such as the pdf,rhos,vels,etc. (Note that in some cases, such as velocity boundary nodes, dynamic variables may remain static throughout simulation. Regardless, the solver treats these variables as dynamic for consistency). In contrast, static variables are pre-computed and defined prior to solving.

Most dynamic variables and associated functions (eg. pdf_eq and calc_pdf_eq) should be implemented within the [`Container`](src/containers.py) class structure which includes [`Cells`](src/cells.py), [`Faces`](src/faces.py), and [`Nodes`](src/nodes.py). The [`src/containers.py`](src/containers.py) file includes base classes for all elements which can be extended by child classes in their respective files (see non-working `CCStencilFaces` class in [`src/faces.py`](src/faces.py)). While it is possible to similarly define a child class of [`Environment`](src/environment.py) to implement dynamic variables and functions, this should be avoided unless necessary for clarity.

In contrast, static variables (eg. stencil cell info, face normals, etc) and its associated functions, should be calculated in the [`Mesher`](src/mesher.py) and passed in through the `to_env()` function. They are similarly stored in each [`Container`](src/containers.py) but treated as a static variable. They may additionally be stored in the [`Mesher`](src/mesher.py) object for future use or redefining the environment.

#### Differentiating between dynamic and static variables
The [`Container`](src/containers.py) class and its subclasses, uses JAX's [pytree](https://docs.jax.dev/en/latest/pytrees.html) class functionality (see [this link](https://docs.jax.dev/en/latest/faq.html#how-to-use-jit-with-methods) and [this link](https://docs.jax.dev/en/latest/jit-compilation.html) for info on why this is necessary) which requires marking all attributes (variables) within a class as dynamic or static using the `tree_flatten()` and  `tree_unflatten()` functions. See below.

```
@jax.tree_util.register_pytree_node_class
class MyClass():
    # Class Variables (should never be dynamic but can be)
    var1
    var2
    def __init__(self,arg1,arg2):
        # Instance Variables (dynamic or static)
        self.var3 = arg1
        self.var4 = arg2

    # JAX flattening and unflattenint
    def tree_flatten(self):
        children = (self.var1,self.var3) # Define as dynamic variables (This data structure is called a tuple)
        aux_data = {'var2': self.var2, 'var4':self.var4} # Define as static variables (This data structure is called a dictionary)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls,aux_data,children):
        obj = cls.__new__(cls)
        obj.var1 = children[0]
        obj.var2 = aux_data['var2']
        obj.var3 = children[1]
        obj.var4 = aux_data['var4']
        return obj
```
It is crucial that all variables defined as static do not change in value throughout simulation as it will trigger recompilation.

All attributes within the Container should be a JAX array with a leading dimension of size M, which represents the number of elements within that container. Then any following array dimensions may be defined, which represents attributes for each individual element. While this is not strictly enforced within the solver, it is crucial to its function and perhaps should be checked.

One consequence of storing an attribute for all elements as one array is that for attributes where each element may differ in size (eg. # of cells for a node) the array must be padded to the length of the largest attribute within any element in the container. This is done in the solver through a custom class `CustomArray` defined in ['utils/utils.py'](utils/utils.py). When `Container.init()` is run, all custom array attributes are converted to raw JAX arrays and loses the padding functionality. Since padding is done with a certain `default_value`, care must be taken to avoid accidental overwriting of data. **For details on its implementation and use, please refer to the source code.**

#### Adding new functionality
All new functionality can either be implemented in an existing [`Container`](src/containers.py) subclass or a new subclass, depending on the need for additional attributes/variables. Additional methods should be implemented and called within these subclasses unless it adds a significant new step in the solver algorithm, which then should be implemented in [`Environment`](src/environment.py) as a new step method. One solution to avoid overcrowding, is to use `env.step()` as a master function with flags which calls additional functions defined in [`Environment`](src/environment.py) depending on the desired solver algorithm. This way, additional solver output or simulation check functionalities (eg. convergence check, animations or intermediate data exporting) can be incorporated cleanly. In this case, `jax.jit` should be placed on the specific solver algorithm methods instead of the master function, `env.step()`.

#### Sharing variables between containers
It is possible to share data between different containers through function calls within the [`Environment`](src/environment.py).


### Meshing and Refinement
both [`src/mesher.py`](src/mesher.py) and [`src/refiner.py`](src/refiner.py) employs significant LLM generated code and should be optimized in the future for readability and reduced computational load. Regardless, the classes within these files should perform as expected.

## Contributors
The list of contributers for the FVDBMSolver is as follows:
- Sungje Park ([email](mailto:sungjepa@usc.edu))
- Leitao Chen ([email](mailto:leitao.chen@erau.edu))
