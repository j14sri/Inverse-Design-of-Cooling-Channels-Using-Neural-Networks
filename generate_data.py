from fenics import *
import numpy as np
import os

def simulate_heat_diffusion(nx=50, ny=50, thermal_load=1.0, save_path="data/"):
    mesh = UnitSquareMesh(nx, ny)
    V = FunctionSpace(mesh, 'P', 1)

    bc = DirichletBC(V, Constant(0), 'on_boundary')
    u = TrialFunction(V)
    v = TestFunction(V)
    f = Constant(thermal_load)
    a = dot(grad(u), grad(v)) * dx
    L = f * v * dx

    u_sol = Function(V)
    solve(a == L, u_sol, bc)

    arr = u_sol.compute_vertex_values(mesh).reshape((nx+1, ny+1))
    os.makedirs(save_path, exist_ok=True)
    np.save(os.path.join(save_path, "temp_map.npy"), arr)

if __name__ == '__main__':
    for t_load in np.linspace(0.5, 2.0, 10):
        simulate_heat_diffusion(thermal_load=t_load, save_path=f"data/t_load_{t_load:.2f}/")
