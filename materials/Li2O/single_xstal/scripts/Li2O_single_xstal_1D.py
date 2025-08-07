import festim as F
import numpy as np
print(F.__version__)

#simulation object
my_model = F.Simulation()

# 1D or 2D mesh representing a uniform domain

#1D:
length = 2e-6  # 2 micrometers
n_cells = 100  # number of cells in the mesh
# Create a 1D mesh
# need volume file for 1D mesh
my_model.mesh = F.MeshFromXDMF(mesh="FESTIM/materials/Li2O/single_xstal/meshes/1D_Test/1d_mesh.xdmf", boundary_file="FESTIM/materials/Li2O/single_xstal/meshes/1D_Test/1d_boundaries.xdmf", volume_file="FESTIM/materials/Li2O/single_xstal/meshes/1D_Test/1d_volumes.xdmf")

#my_mesh_1D = F.MeshFromRefinement(Geometry=F.Interval(0, length), n_cells=n_cells)

#materials
#Lithium oxide (Li2O)
Li2O = F.CppCustomMaterial(
            id=1,
            D="1e-7 * (1 + 1e-20 * c_0) * exp(-0.4 / (8.617e-5 * T))",  # m²/s
            S="1e22",  # constant solubility for simplicity (H/m³/Pa^0.5)
            traps=[],
            )
# Li2O = F.Material(
#     name = "Li2O",
#     id=1,
#     D_0=1.03613E-6, # Diffusion coefficient at 300 K
#     E_D=0.9028, # Activation energy for diffusion
#     thermal_cond=6.5, #6-7RTP 5-10 @200-900 C
#     heat_capacity=2049, #25 degrees C
#     rho=2.013e3, #25 degrees C

# )

#Lithium carbide (Li2C2)
#Note: The values for Li2C2 are hypothetical and should be replaced with accurate data
# Li2C2 = F.Material(
#     name = "Li2C2",
#     id=2,
#     D_0=0,
#     E_D=0,
#     thermal_conductivity=0,
#     heat_capacity=0,
#     rho=1.300e3,
# )

#Lithium Titanate (Li2TiO3)
#Note: The values for Li2TiO3 are hypothetical and should be replaced with accurate data
# Li2TiO3 = F.Material(
#     name = "Li2TiO3",
#     id=3,
#     D_0=0,
#     E_D=0,
#     thermal_conductivity=3.0, #300 K
#     heat_capacity=0,
#     rho=3.430e3,
# )

#Lithium Orthosilicate (Li4SiO4)
# Li4SiO4 = F.Material(
#     name = "Li4SiO4",
#     id=4,
#     D_0=0,
#     E_D=0,
#     thermal_conductivity=4.8, #RT 2.8@1100 K
#     heat_capacity=931,
#     rho=2.0280e3,
# )

my_model.materials = [Li2O]  # Add other materials as needed

#temperature
# ~RT
my_model.T = 300

#must define a source of Tritium intial condition of Tritium concentration e.g. 1e15 H/m3
#boundary conditions simulatin T release from the surface (desorption)
#zero concentration boundary mimics a perfect vacuum sink
#two surfaces are defined for the boundary conditions
my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[1, 2], value=1e15, field=0)  # H/m3
]

#source of tritium
my_model.sources = [F.Source(value=1e20, volume=1, field=0)]  # H/m3/s

#settings
my_model.settings = F.Settings(
    absolute_tolerance=1e10, relative_tolerance=1e-10, final_time=100  # s
)

#exports
results_folder = "Li2O/1D_results"
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

