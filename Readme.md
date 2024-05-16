### Soft Body Dataset Generation for Mujoco


1. **Generation Tetrahedal Mesh**:
We start with a triangle-mesh called for e.g. `mesh.obj` and generate a tetrahedal mesh using the `fTetWild` library, [here](https://github.com/wildmeshing/fTetWild/tree/master?tab=readme-ov-file). The tetrahedal mesh is saved in a file called `mesh.msh`. 


2. **Export as GMSH v4.1 format**:
Mujoco supporst only the GMSH file format 4.1. Download and extract the gmsh software from [here](https://gmsh.info/#Download). Double click on the `bin/gmsh` executable and open the `mesh.msh` file. Export the mesh as `mesh_gmsh.msh` in the GMSH v4.1 format. 

3. **Generate Mujoco dataset**:
Run the `generate_dataset.py` script to generate the mujoco dataset. The script will generate the mujoco dataset in the `dataset` folder.


