import festim as F 
import numpy as np
print(F.__version__)

#simulation object
my_model = F.Simulation()

#mesh
F.MeshFromVertices(vertices=[0, 1, 2, 3, 4, 5, 6, 7, 7.5])
my_model.mesh = F.MeshFromVertices(vertices=np.linspace(0, 7e-6, num=1001))

#materials 
my_model.materials = F.Material(id=1, D_0=1e-7, E_D=0.2)

#temperature 
my_model.T = 300 

#boundary conditions
my_model.boundary_conditions = [
    F.DirichletBC(surfaces=[1, 2], value=1e15, field=0)  # H/m3
]


my_model.sources = [F.Source(value=1e20, volume=1, field=0)]  # H/m3/s

#settings
my_model.settings = F.Settings(
    absolute_tolerance=1e10, relative_tolerance=1e-10, final_time=2  # s
)

#exports
results_folder = "/home/tmg25bcx/FESITM/task01"
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

#stepsize
my_model.dt = F.Stepsize(0.05, milestones=[0.1, 0.2, 0.5, 1])  

#solve steady-state
my_model.settings.transient = False
my_model.dt = None

my_model.exports = [
    F.TXTExport(
        field="solute", filename=results_folder + "/mobile_concentration_steady.txt"
    )
]

#run
my_model.initialise()

my_model.run()