from firedrake import *
import pytest
from petsc4py import PETSc
from pyop2.mpi import COMM_WORLD
from pyop2 import RW
import os
from os.path import abspath, dirname, join

cwd = abspath(dirname(__file__))


def _get_expr(mesh):
   dim = mesh.topological_dimension()
   if dim == 2:
        x, y = SpatialCoordinate(mesh)
        return x * y * y
   elif dim == 3:
        x, y, z = SpatialCoordinate(mesh)
        return x * y * y * z * z * z
   else:
        raise ValueError(f"Not expecting a {dim}-dimensional mesh")


@pytest.mark.parallel(nprocs=4)
@pytest.mark.parametrize('cell_family_degree', [("triangle", "P", 1),
                                                ("triangle", "P", 2),
                                                ("triangle", "P", 5),
                                                ("triangle", "DP", 0),
                                                ("triangle", "DP", 1),
                                                ("triangle", "DP", 2),
                                                ("triangle", "DP", 6),
                                                ("tetrahedra", "P", 1),
                                                ("tetrahedra", "P", 2),
                                                ("tetrahedra", "P", 3),
                                                ("tetrahedra", "P", 6),
                                                ("tetrahedra", "DP", 0),
                                                ("tetrahedra", "DP", 1),
                                                ("tetrahedra", "DP", 2),
                                                ("tetrahedra", "DP", 3),
                                                ("tetrahedra", "DP", 7),
                                                ("quad", "Q", 1),
                                                ("quad", "Q", 2),
                                                ("quad", "Q", 5),
                                                ("quad", "DQ", 0),
                                                ("quad", "DQ", 1),
                                                ("quad", "DQ", 2),
                                                ("quad", "DQ", 6),])
def test_io_function_simplex(cell_family_degree, tmpdir):
    # Parameters
    cell_type, family, degree = cell_family_degree

    comm = COMM_WORLD
    filename = os.path.join(str(tmpdir), "test_io_function_dump.h5")
    filename = COMM_WORLD.bcast(filename, root=0)
    ntimes = 3
    meshname = "exampleDMPlex"
    fs_name = "example_function_space"
    func_name = "example_function"
    if cell_type == "triangle":
        meshA = Mesh("./docs/notebooks/stokes-control.msh", name=meshname, comm=comm)
    elif cell_type == "tetrahedra":
        meshA = Mesh(join(cwd, "..", "meshes", "sphere.msh"),
                     name=meshname, comm=comm)
    elif cell_type == "quad":
        meshA = Mesh(join(cwd, "..", "meshes", "unitsquare_unstructured_quadrilaterals.msh"),
                     name=meshname, comm=comm)
    meshA.init()
    meshA.name = meshname
    plexA = meshA.topology.topology_dm
    VA = FunctionSpace(meshA, family, degree, name=fs_name)
    expr = _get_expr(meshA)
    fA = Function(VA, name=func_name)
    fA.interpolate(expr)
    with CheckpointFile(filename, 'w', comm=comm) as afile:
        afile.save_function(fA)
    volA = assemble(fA * dx)
    # Load -> View cycle
    grank = COMM_WORLD.rank
    for i in range(ntimes):
        mycolor = (grank > ntimes - i)
        comm = COMM_WORLD.Split(color=mycolor, key=grank)
        if mycolor == 0:
            # Load
            with CheckpointFile(filename, "r", comm=comm) as afile:
                fB = afile.load_function(func_name, mesh_name=meshname)
            # Check
            volB = assemble(fB * dx)
            assert abs(volB - volA) < 1.e-7
            VB = fB.function_space()
            expr = _get_expr(VB.mesh())
            fBe = Function(VB).interpolate(expr)
            assert assemble(inner(fB - fBe, fB - fBe) * dx) < 1.e-16
            # Save
            with CheckpointFile(filename, 'w', comm=comm) as afile:
                afile.save_function(fB)


if __name__ == "__main__":
    test_io_function_simplex(("tetrahedra", "DG", 5), "./")
