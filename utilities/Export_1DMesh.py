import festim as F
import dolfin as df
import meshio

# --- 1. Create the 1D mesh ---
length = 2e-6  # 1 mm
n_cells = 100

mesh = df.IntervalMesh(n_cells, 0, length)

# --- 2. Mark the boundary surfaces ---
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 0.0) and on_boundary

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], length) and on_boundary

left = Left()
right = Right()

left.mark(boundaries, 1)   # surface ID 1
right.mark(boundaries, 2)  # surface ID 2

# --- 3. Mark the volume (cells) ---
volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 1)  # All cells marked as 1


# --- 4. Save the mesh and boundaries to XDMF ---
with df.XDMFFile("meshes/1D_Test/1d_mesh.xdmf") as mesh_file:
    mesh_file.write(mesh)

with df.XDMFFile("meshes/1D_Test/1d_boundaries.xdmf") as facet_file:
    facet_file.write(boundaries)

with df.XDMFFile("meshes/1D_Test/1d_volumes.xdmf") as volume_file:
    volume_file.write(volumes)
