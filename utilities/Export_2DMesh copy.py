import dolfin as df
import os

# --- 1. Create the 2D mesh ---
width = 1e-3   # 1 mm
height = 1e-3  # 1 mm
n_cells = 50   # number of cells along each axis

mesh = df.RectangleMesh(
    df.Point(0.0, 0.0),
    df.Point(width, height),
    n_cells,
    n_cells
)

# --- 2. Mark boundary surfaces with unique IDs ---
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], 0.0) and on_boundary

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], width) and on_boundary

class Bottom(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], 0.0) and on_boundary

class Top(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[1], height) and on_boundary

# Create and mark subdomains
Left().mark(boundaries, 1)
Right().mark(boundaries, 2)
Bottom().mark(boundaries, 3)
Top().mark(boundaries, 4)

# --- 3. Mark volumes ---
volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 1)  # All cells marked as 1


# --- 4. Export to XDMF ---
output_dir = "meshes/2D_Test"
os.makedirs(output_dir, exist_ok=True)

# --- 3. Export mesh and boundary markers to XDMF ---
with df.XDMFFile(f"{output_dir}/2d_vertices_mesh.xdmf") as mesh_file:
    mesh_file.write(mesh)

with df.XDMFFile(f"{output_dir}/2d__vertices_boundaries.xdmf") as boundary_file:
    boundary_file.write(boundaries)

with df.XDMFFile(f"{output_dir}/2D_vertices_volumes.xdmf") as volume_file:
    volume_file.write(volumes)
