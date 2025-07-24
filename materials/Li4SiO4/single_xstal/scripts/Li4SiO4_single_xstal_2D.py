import festim as F
import numpy as np
print(F.__version__)

#simulation object
my_model = F.Simulation()

# 1D or 2D mesh representing a uniform domain

# Load 1D mesh from XDMF
# need volume file for 1D mesh
my_model.mesh = F.MeshFromXDMF(mesh="meshes/2D_Test/2d_vertices_mesh.xdmf", boundary_file="meshes/2D_Test/2d__vertices_boundaries.xdmf", volume_file="meshes/2D_Test/2D_vertices_volumes.xdmf")

#my_mesh_1D = F.MeshFromRefinement(Geometry=F.Interval(0, length), n_cells=n_cells)

#materials
#Lithium Orthosilicate (Li4SiO4)
Li4SiO4 = F.Material(
    name = "Li4SiO4",
    id=4,
    D_0=0,
    E_D=0,
    thermal_conductivity=4.8, #RT 2.8@1100 K
    heat_capacity=931,
    rho=2.0280e3,
)

my_model.materials = [Li4SiO4]  # Add other materials as needed

#temperature
# ~RT
my_model.T = 300

#must define a source of Tritium intial condition of Tritium concentration e.g. 1e15 H/m3
#boundary conditions simulatin T release from the surface (desorption)
#zero concentration boundary mimics a perfect vacuum sink
#two surfaces are defined for the boundary conditions
my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[1, 2, 3 , 4], value=1e15, field=0)  # H/m3
]

#source of tritium
my_model.sources = [F.Source(value=1e20, volume=1, field=0)]  # H/m3/s

#settings
my_model.settings = F.Settings(
    absolute_tolerance=1e10, relative_tolerance=1e-10, final_time=100  # s
)

#exports
results_folder = "Li2O/2D_results"
my_model.exports = [
    F.XDMFExport(
        field="solute",
        filename=results_folder + "/hydrogen_concentration.xdmf",
        checkpoint=False,  # needed in 1D
    ),
    F.TXTExport(
        field="solute",
        times=[0.1, 0.2, 0.5, 1],
        filename=results_folder + "/mobile_concentration.txt",
    ),
]

my_model.dt = F.Stepsize(0.05, milestones=[0.1, 0.2, 0.5, 1])  # s

my_model.initialise()

my_model.run()