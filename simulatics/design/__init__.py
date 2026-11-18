"""Robot design tools: parametric CAD links, simulation loads, topology optimisation and design search.

  cad.py        HumanoidParams -> hollow-shell CAD parts (manifold3d), mass properties (trimesh),
                STL/OBJ/GLB/STEP export, MJCF with mesh visuals and CAD inertials
  urdf.py       URDF export (kinematics from the compiled model; CAD or MuJoCo inertials)
  topopt.py     3D SIMP topology optimisation (hex8 voxels, density filter, OC; C++ multigrid/Jacobi PCG)
  loads.py      per-link joint reaction wrenches from simulated walking / standing with pushes
  linkopt.py    optimize_link: topology-optimise one limb against its simulated load cases
  evaluate.py   scoring a body design by simulating it (walk to goals, or stand with pushes)
  optimizer.py  DesignOptimizer: CMA-ES or Gaussian-process Bayesian optimisation over body genes

Submodules are imported on demand (`from simulatics.design import cad`) so importing the package
stays cheap.
"""
