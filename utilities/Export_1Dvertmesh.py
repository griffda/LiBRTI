import numpy as np
import dolfin as df
import os

#mesh
F.MeshFromVertices(vertices=[0, 1, 2, 3, 4, 5, 6, 7, 7.5])
my_model.mesh = F.MeshFromVertices(vertices=np.linspace(0, 7e-6, num=1001))

# --- 1. Define vertices and mesh ---
vertices = np.linspace(0, 7e-6, num=1001)
n_cells = len(vertices) - 1
mesh = df.IntervalMesh(n_cells, vertices[0], vertices[-1])

# --- 2. Mark boundaries ---
boundaries = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)

class Left(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], vertices[0]) and on_boundary

class Right(df.SubDomain):
    def inside(self, x, on_boundary):
        return df.near(x[0], vertices[-1]) and on_boundary

left = Left()
right = Right()
left.mark(boundaries, 1)
right.mark(boundaries, 2)

# --- 3. Mark volumes ---
volumes = df.MeshFunction("size_t", mesh, mesh.topology().dim(), 1)  # All cells marked as 1

# --- 4. Export to XDMF ---
output_dir = "meshes/vertices_mesh"
os.makedirs(output_dir, exist_ok=True)

with df.XDMFFile(f"{output_dir}/mesh.xdmf") as mesh_file:
    mesh_file.write(mesh)

with df.XDMFFile(f"{output_dir}/boundaries.xdmf") as facet_file:
    facet_file.write(boundaries)

with df.XDMFFile(f"{output_dir}/volumes.xdmf") as volume_file:
    volume_file.write(volumes)

print("Mesh, boundaries, and volumes exported to", output_dir)