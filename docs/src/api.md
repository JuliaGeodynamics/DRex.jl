# API Reference

## Mineral types and parameters

```@docs
Mineral
MineralPhase
MineralFabric
DeformationRegime
DefaultParams
default_params
StiffnessTensors
```

## CPO integration

```@docs
update_orientations!
update_all!
run_pathlines_batch!
derivatives!
get_crss
```

## Voigt averaging

```@docs
voigt_averages
voigt_decompose
voigt_to_elastic_tensor
elastic_tensor_to_voigt
voigt_matrix_to_vector
voigt_vector_to_matrix
rotate_tensor
polar_decompose
upper_tri_to_symmetric
```

## Diagnostics

```@docs
misorientation_index
bingham_average
finite_strain
symmetry_pgr
elasticity_components
coaxial_index
smallest_angle
```

## Velocity fields

```@docs
simple_shear_2d
cell_2d
corner_2d
get_pathline
```

## Geometry and projections

```@docs
poles
lambert_equal_area
to_cartesian
to_spherical
```

## Statistics and resampling

```@docs
resample_orientations
misorientation_angles
misorientation_hist
misorientations_random
```

## Utilities

```@docs
strain_increment
apply_gbs!
```

## LaMEM extension

These are available after `using LaMEM, GeophysicalModelGenerator, WriteVTK`.

```@docs
LaMEMSnapshot
CPOTracer
compute_cpo_from_lamem
load_snapshots
create_tracers
backtrack_positions
evolve_cpo!
run_cpo_at_locations
trilinear_interpolate
interpolate_vel_grad
interpolate_velocity
make_velocity_gradient_func
make_velocity_func
```

## SCSV file I/O

```@docs
read_scsv
save_scsv
write_scsv_header
```
