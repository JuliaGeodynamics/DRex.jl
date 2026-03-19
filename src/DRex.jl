"""
    DRex

Simulate crystallographic preferred orientation (CPO) evolution in polycrystals.

This is a Julia translation of the Python PyDRex package, with key routines designed
to be allocation-free using StaticArrays.

# Core types
- `MineralPhase`: olivine, enstatite
- `MineralFabric`: olivine_A through olivine_E, enstatite_AB
- `DeformationRegime`: min_viscosity through max_viscosity
- `DefaultParams`: default simulation parameters

# Key functions
- `derivatives!`: core D-Rex solver (allocation-free)
- `voigt_averages`: elastic tensor averaging
- `misorientation_index`: texture strength diagnostic

# Modules
- `Core`: enums, parameters, core solver
- `Tensors`: tensor operations, Voigt notation
- `Geometry`: coordinate conversions, projections
- `Minerals`: mineral class, stiffness tensors
- `Diagnostics`: texture diagnostics
- `Velocity`: analytical velocity fields
- `Stats`: statistical methods
- `Utils`: utility functions
"""
module DRex

using LinearAlgebra
using StaticArrays
using Random
using Rotations

include("core.jl")
include("tensors.jl")
include("geometry.jl")
include("utils.jl")
include("stats.jl")
include("minerals.jl")
include("diagnostics.jl")
include("io.jl")
include("velocity.jl")
include("pathlines.jl")
include("lamem_coupling.jl")

export MineralPhase, olivine, enstatite
export MineralFabric, olivine_A, olivine_B, olivine_C, olivine_D, olivine_E, enstatite_AB
export DeformationRegime, min_viscosity, matrix_diffusion, boundary_diffusion,
       sliding_diffusion, matrix_dislocation, sliding_dislocation,
       frictional_yielding, max_viscosity
export DefaultParams, default_params
export derivatives!, get_crss
export Mineral, update_orientations!, update_all!, StiffnessTensors, voigt_averages
export OLIVINE_SLIP_SYSTEMS, OLIVINE_PRIMARY_AXIS

# Diagnostics
export elasticity_components, bingham_average, finite_strain
export symmetry_pgr, misorientation_index, coaxial_index, smallest_angle

# Velocity & Pathlines
export simple_shear_2d, cell_2d, corner_2d
export get_pathline

# Geometry
export LatticeSystem, orthorhombic, monoclinic, triclinic, tetragonal, hexagonal
export to_cartesian, to_spherical, misorientation_angles
export poles, lambert_equal_area, shirley_concentric_squaredisk, to_indices2d

# Tensors
export voigt_decompose, voigt_to_elastic_tensor, elastic_tensor_to_voigt
export voigt_matrix_to_vector, voigt_vector_to_matrix
export rotate_tensor, polar_decompose, upper_tri_to_symmetric

# Stats
export resample_orientations, misorientation_hist, misorientations_random

# Utils
export strain_increment, apply_gbs!, extract_vars, remove_nans, default_ncpus, add_dim

# IO
export SCSVError, read_scsv, save_scsv, write_scsv_header, scsv_data

# LaMEM coupling (implemented in ext/DRexLaMEMExt.jl when LaMEM is loaded)
export LaMEMSnapshot, CPOTracer
export load_snapshots, create_tracers, evolve_cpo!
export backtrack_positions, run_cpo_at_locations, compute_cpo_from_lamem
export trilinear_interpolate, interpolate_vel_grad, interpolate_velocity
export make_velocity_gradient_func, make_velocity_func

end # module
