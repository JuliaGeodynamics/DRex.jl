#=
# run_lamem_subduction3d.jl
#
# Set up and run the 3D subduction LaMEM simulation from Schellart et al. (2007).
# Outputs the velocity and velocity gradient tensor fields required for CPO
# post-processing with postprocess_cpo.jl.
#
# Usage:
#   cd examples/LaMEM
#   julia --project=. run_lamem_subduction3d.jl
#
# Output directory: Subduction_3D_CPO/
=#

using LaMEM, GeophysicalModelGenerator

# ── Grid and solver ──────────────────────────────────────────────────────────

model = Model(
    Grid(
        #nel = (64, 16, 32),    # lower resolution for better CI integration
        nel = (128, 32, 64),  
        x   = [-3960, 500],
        y   = [0, 2640],
        z   = [-660, 0],
    ),

    BoundaryConditions(noslip=[0, 0, 0, 0, 1, 0]),

    Solver(
        SolverType    = "multigrid",
        MGLevels      = 4,
        MGCoarseSolver = Sys.iswindows() ? "direct" : "mumps",
        PETSc_options = [
            "-snes_type ksponly",
            "-js_ksp_rtol 1e-3",
            "-js_ksp_atol 1e-4",
            "-js_ksp_monitor",
        ],
    ),

    # Enable velocity gradient tensor — required for DRex CPO coupling
    Output(
        out_file_name     = "Subduction_3D_CPO",
        out_dir           = "Subduction_3D_CPO",
        out_vel_gr_tensor = 1,
        out_velocity      = 1,
    ),

    Time(
        nstep_max = 200,
        nstep_out = 1,       # save every step so the CPO post-processor has full resolution
        time_end  = 100,
        dt_min    = 1e-5,
    ),

    Scaling(GEO_units(length=1km, stress=1e9Pa)),
)

# ── Geometry ─────────────────────────────────────────────────────────────────

# Horizontal slab (crust + lithospheric mantle)
add_box!(model,
    xlim  = (-3000, -1000), ylim = (0, 1000), zlim = (-80, 0),
    phase = LithosphericPhases(Layers=[20, 60], Phases=[1, 2]),
)

# Inclined slab tip
add_box!(model,
    xlim     = (-1000, -810), ylim = (0, 1000), zlim = (-80, 0),
    phase    = LithosphericPhases(Layers=[20, 60], Phases=[1, 2]),
    DipAngle = 16,
)

# ── Material properties ───────────────────────────────────────────────────────

add_phase!(model,
    Phase(Name="mantle", ID=0, eta=1e21,  rho=3200),
    Phase(Name="crust",  ID=1, eta=1e21,  rho=3280),
    Phase(Name="slab",   ID=2, eta=2e23,  rho=3280),
)

# ── Run ───────────────────────────────────────────────────────────────────────

n_cores = Sys.iswindows() ? 1 : 8
@info "Running LaMEM on $n_cores core(s) → Subduction_3D_CPO/"
run_lamem(model, n_cores)
@info "LaMEM run complete."
