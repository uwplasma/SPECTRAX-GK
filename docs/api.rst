API Reference
=============

Public API Registry
-------------------

The top-level ``spectraxgk`` package remains the stable user-facing facade.
Implementation ownership lives in the domain modules below; the API registry
only groups and re-exports those symbols so the public surface is auditable
without mixing physics kernels, validation gates, and executable workflows in a
single large ``__init__.py`` file.

.. automodule:: spectraxgk.api
   :members:
   :exclude-members: KrylovConfig
   :no-index:

.. automodule:: spectraxgk.api.configuration
   :members:
   :no-index:

.. automodule:: spectraxgk.api.geometry
   :members:
   :no-index:

.. automodule:: spectraxgk.api.diagnostics
   :members:
   :no-index:

.. automodule:: spectraxgk.api.runtime
   :members:
   :no-index:

.. automodule:: spectraxgk.api.solvers
   :members:
   :exclude-members: KrylovConfig
   :no-index:

.. automodule:: spectraxgk.api.benchmarks
   :members:
   :no-index:

.. automodule:: spectraxgk.api.validation
   :members:
   :no-index:

.. automodule:: spectraxgk.api.parallel
   :members:
   :no-index:

.. automodule:: spectraxgk.api.objectives
   :members:
   :no-index:

.. automodule:: spectraxgk.api.artifacts
   :members:
   :no-index:

Core Refactor Contracts
-----------------------

.. automodule:: spectraxgk.core.contracts
   :members:

Core Extension Points
---------------------

.. automodule:: spectraxgk.core.extension_points
   :members:

Velocity-Space Core
-------------------

.. automodule:: spectraxgk.core.velocity
   :members:

Geometry
--------

.. automodule:: spectraxgk.geometry
   :members:
   :no-index:

Geometry Core
-------------

.. automodule:: spectraxgk.geometry.core
   :members:
   :private-members:

Geometry Boundaries
-------------------

.. automodule:: spectraxgk.geometry.boundaries
   :members:

Analytic Geometry
-----------------

.. automodule:: spectraxgk.geometry.analytic
   :members:
   :private-members:

Flux-Tube Geometry
------------------

.. automodule:: spectraxgk.geometry.flux_tube
   :members:
   :private-members:

Miller EIK Generation
---------------------

.. automodule:: spectraxgk.geometry.miller_eik
   :members:
   :private-members:

VMEC EIK Generation
-------------------

.. automodule:: spectraxgk.geometry.vmec_eik
   :members:
   :private-members:

Differentiable Geometry
-----------------------

.. automodule:: spectraxgk.geometry.differentiable
   :members:

Differentiable Geometry Backend Discovery
-----------------------------------------

.. automodule:: spectraxgk.geometry.backend_discovery
   :members:
   :private-members:

Differentiable Flux-Tube Contract
---------------------------------

.. automodule:: spectraxgk.geometry.flux_tube_contract
   :members:
   :private-members:

Differentiable Geometry AD Checks
---------------------------------

.. automodule:: spectraxgk.geometry.autodiff_checks
   :members:
   :private-members:

Differentiable Geometry Sensitivity
-----------------------------------

.. automodule:: spectraxgk.geometry.sensitivity
   :members:
   :private-members:

Differentiable Boozer Bridge
----------------------------

.. automodule:: spectraxgk.geometry.booz_xform_bridge
   :members:
   :private-members:

Differentiable VMEC-State Sensitivity
-------------------------------------

.. automodule:: spectraxgk.geometry.vmec_state_sensitivity
   :members:
   :private-members:

Differentiable VMEC Boozer Core
-------------------------------

.. automodule:: spectraxgk.geometry.vmec_boozer_core
   :members:
   :private-members:

Differentiable VMEC Boozer Constants
------------------------------------

.. automodule:: spectraxgk.geometry.vmec_boozer_constants
   :members:
   :private-members:

Differentiable VMEC Flux-Tube Reports
-------------------------------------

.. automodule:: spectraxgk.geometry.vmec_flux_tube_reports
   :members:
   :private-members:

Differentiable VMEC Tensor Mapping
----------------------------------

.. automodule:: spectraxgk.geometry.vmec_tensor_mapping
   :members:
   :private-members:

Differentiable Geometry Numerics
--------------------------------

.. automodule:: spectraxgk.geometry.numerics
   :members:
   :private-members:

Grids
-----

.. automodule:: spectraxgk.core.grid
   :members:

Species
-------

.. automodule:: spectraxgk.core.species
   :members:

Operators
---------

.. automodule:: spectraxgk.operators
   :members:

Linear Operators
----------------

.. automodule:: spectraxgk.operators.linear

Linear
------

.. automodule:: spectraxgk.linear
   :members:

Linear Linked Boundaries
------------------------

.. automodule:: spectraxgk.operators.linear.linked
   :members:
   :private-members:

Linear Cache
------------

.. automodule:: spectraxgk.operators.linear.cache
   :members:
   :private-members:

.. automodule:: spectraxgk.operators.linear.cache_model
   :members:
   :no-index:

.. automodule:: spectraxgk.operators.linear.cache_arrays
   :members:
   :private-members:
   :no-index:

.. automodule:: spectraxgk.operators.linear.cache_builder
   :members:
   :no-index:

Linear Moments
--------------

.. automodule:: spectraxgk.operators.linear.moments
   :members:
   :private-members:

Linear Parameters
-----------------

.. automodule:: spectraxgk.operators.linear.params
   :members:
   :private-members:

Linear RHS
----------

.. automodule:: spectraxgk.operators.linear.rhs
   :members:

Linear Implicit Solvers
-----------------------

.. automodule:: spectraxgk.solvers.linear.implicit
   :members:
   :private-members:

Linear Integrators
------------------

.. automodule:: spectraxgk.solvers.linear.integrators
   :members:
   :private-members:

Linear Diagnostic Integration
-----------------------------

.. automodule:: spectraxgk.solvers.linear.integrator_diagnostics
   :members:
   :private-members:

Linear Parallel RHS
-------------------

.. automodule:: spectraxgk.solvers.linear.parallel
   :members:
   :private-members:

Linear Parallel Policy
----------------------

.. automodule:: spectraxgk.solvers.linear.parallel_common
   :members:
   :private-members:

Linear Parallel Streaming
-------------------------

.. automodule:: spectraxgk.solvers.linear.parallel_streaming
   :members:
   :private-members:

Linear Parallel Electrostatic Slices
------------------------------------

.. automodule:: spectraxgk.solvers.linear.parallel_electrostatic
   :members:
   :private-members:

Linear Krylov Solvers
---------------------

.. automodule:: spectraxgk.solvers.linear.krylov
   :members:
   :private-members:

Linear Eigenmode Solver Internals
---------------------------------

``spectraxgk.solvers.linear.eigen_policy`` owns ``KrylovConfig``; the public
``spectraxgk.solvers.linear.krylov`` facade re-exports it so existing scripts
and TOML loaders keep the same import path.

.. automodule:: spectraxgk.solvers.linear.eigen_operator
   :members:
   :private-members:
   :no-index:

.. automodule:: spectraxgk.solvers.linear.eigen_selection
   :members:
   :private-members:
   :no-index:

.. automodule:: spectraxgk.solvers.linear.eigen_preconditioners
   :members:
   :private-members:
   :no-index:

.. automodule:: spectraxgk.solvers.linear.krylov_algorithms
   :members:
   :private-members:
   :no-index:

Nonlinear Diagnostics
---------------------

.. automodule:: spectraxgk.operators.nonlinear.diagnostics
   :members:
   :private-members:

Nonlinear Diagnostic State
--------------------------

.. automodule:: spectraxgk.operators.nonlinear.diagnostic_state
   :members:
   :private-members:

Nonlinear Collision Split Helpers
---------------------------------

.. automodule:: spectraxgk.operators.nonlinear.collisions
   :members:
   :private-members:

Nonlinear Helpers
-----------------

.. automodule:: spectraxgk.operators.nonlinear.policies
   :members:
   :private-members:

Nonlinear Projection Helpers
----------------------------

.. automodule:: spectraxgk.operators.nonlinear.projection
   :members:
   :private-members:

Nonlinear RHS
-------------

.. automodule:: spectraxgk.operators.nonlinear.rhs
   :members:
   :private-members:

Nonlinear Bracket Kernels
-------------------------

.. automodule:: spectraxgk.terms.brackets
   :members:
   :private-members:

Nonlinear Gyroaveraging
-----------------------

.. automodule:: spectraxgk.terms.gyroaveraging
   :members:
   :private-members:

Nonlinear Term Assembly
-----------------------

.. automodule:: spectraxgk.terms.nonlinear
   :members:
   :private-members:

Reduced cETG Model
------------------

.. automodule:: spectraxgk.terms.reduced

.. automodule:: spectraxgk.terms.reduced.cetg_model
   :members:

.. automodule:: spectraxgk.terms.reduced.cetg_state
   :members:
   :private-members:

.. automodule:: spectraxgk.terms.reduced.cetg_rhs
   :members:
   :private-members:

.. automodule:: spectraxgk.terms.reduced.cetg_integrator
   :members:
   :private-members:

Nonlinear Explicit Step
-----------------------

.. automodule:: spectraxgk.solvers.nonlinear.explicit
   :members:
   :private-members:

Nonlinear State Integration
---------------------------

.. automodule:: spectraxgk.solvers.nonlinear.state_integration
   :members:
   :private-members:

Nonlinear Diagnostic Drivers
----------------------------

.. automodule:: spectraxgk.solvers.nonlinear.diagnostics
   :members:
   :private-members:

Nonlinear Diagnostic Integration
--------------------------------

.. automodule:: spectraxgk.solvers.nonlinear.diagnostic_integration
   :members:
   :private-members:

Nonlinear IMEX
--------------

.. automodule:: spectraxgk.solvers.nonlinear.imex
   :members:
   :private-members:

Explicit Time Integrators
-------------------------

.. automodule:: spectraxgk.solvers.time.explicit
   :members:
   :private-members:

Explicit Step Kernels
---------------------

.. automodule:: spectraxgk.solvers.time.explicit_steps
   :members:
   :private-members:

Explicit Diagnostic Integrators
-------------------------------

.. automodule:: spectraxgk.solvers.time.explicit_diagnostics
   :members:
   :private-members:

Explicit CFL Policy
-------------------

.. automodule:: spectraxgk.solvers.time.explicit_cfl
   :members:
   :private-members:

Explicit Progress Helpers
-------------------------

.. automodule:: spectraxgk.solvers.time.explicit_progress
   :members:
   :private-members:

Diffrax Time Integrators
------------------------

.. automodule:: spectraxgk.solvers.time.diffrax
   :members:
   :private-members:

Config-Driven Time Runners
--------------------------

.. automodule:: spectraxgk.solvers.time.runners
   :members:
   :private-members:

Runtime Execution Dispatch
--------------------------

.. automodule:: spectraxgk.workflows.runtime.execution
   :members:
   :private-members:

Nonlinear Replicate Diagnostics
-------------------------------

.. automodule:: spectraxgk.validation.nonlinear_transport.replicate_diagnostics
   :members:

Nonlinear Replicate Follow-Up
-----------------------------

.. automodule:: spectraxgk.validation.nonlinear_transport.replicate_followup
   :members:

Production Nonlinear Optimization Guard
---------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_transport.optimization_guard
   :members:

.. automodule:: spectraxgk.validation.nonlinear_transport.optimization_policy
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.nonlinear_transport.optimization_reports
   :members:
   :private-members:

Nonlinear Gradient Follow-Up Core
---------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_core
   :members:
   :private-members:

Nonlinear Gradient Follow-Up Variance
-------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_variance
   :members:
   :private-members:

Nonlinear Gradient Follow-Up Candidate Design
---------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_candidate
   :members:
   :private-members:

Nonlinear Gradient Follow-Up Composite Controls
-----------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_composite
   :members:
   :private-members:

Nonlinear Gradient Follow-Up Matched Plans
------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_plan
   :members:
   :private-members:

Nonlinear Gradient Follow-Up QL Seed Screens
--------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_ql_seed
   :members:
   :private-members:

Nonlinear Gradient Follow-Up State Runbooks
-------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.followup_state_runbook
   :members:
   :private-members:

Nonlinear Gradient Evidence Core
--------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.evidence_core
   :members:
   :private-members:

Nonlinear Gradient Evidence Classification
------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.evidence_classification
   :members:
   :private-members:

Nonlinear Gradient Evidence Windows
-----------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.evidence_windows
   :members:
   :private-members:

Nonlinear Gradient Evidence Finite Difference
---------------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.evidence_fd
   :members:
   :private-members:

Nonlinear Gradient Evidence Screening
-------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.evidence_screening
   :members:
   :private-members:

Nonlinear Gradient Evidence Gap Reports
---------------------------------------

.. automodule:: spectraxgk.validation.nonlinear_gradient.evidence_gap
   :members:
   :private-members:

Benchmarks
----------

.. automodule:: spectraxgk.benchmarks
   :members:

Benchmark Initialization
------------------------

.. automodule:: spectraxgk.validation.benchmarks.initialization
   :members:
   :private-members:

Benchmark Fit Signals
---------------------

.. automodule:: spectraxgk.validation.benchmarks.fit_signals
   :members:
   :private-members:

Benchmark Batching
------------------

.. automodule:: spectraxgk.validation.benchmarks.batching
   :members:
   :private-members:

Benchmark Solver Policies
-------------------------

.. automodule:: spectraxgk.validation.benchmarks.solver_policy
   :members:
   :private-members:

Benchmark Reference Data
------------------------

.. automodule:: spectraxgk.validation.benchmarks.reference
   :members:
   :private-members:

Benchmark Species Policies
--------------------------

.. automodule:: spectraxgk.validation.benchmarks.species
   :members:
   :private-members:

Benchmark Defaults
------------------

.. automodule:: spectraxgk.validation.benchmarks.defaults
   :members:

Benchmark Scan Policies
-----------------------

.. automodule:: spectraxgk.validation.benchmarks.scan
   :members:
   :private-members:

Benchmark Secondary Workflow
----------------------------

.. automodule:: spectraxgk.validation.benchmarks.secondary
   :members:
   :private-members:

Benchmarking
------------

.. automodule:: spectraxgk.validation.benchmarks.harness
   :members:
   :no-index:

.. automodule:: spectraxgk.validation.benchmarks.harness_eigenfunctions
   :members:

.. automodule:: spectraxgk.validation.benchmarks.harness_timeseries
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.harness_metrics
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.harness_zonal_metrics
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.harness_scan
   :members:

.. automodule:: spectraxgk.validation.benchmarks.cyclone
   :members:
   :no-index:

.. automodule:: spectraxgk.validation.benchmarks.cyclone_linear
   :members:

.. automodule:: spectraxgk.validation.benchmarks.cyclone_linear_paths
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.cyclone_scan
   :members:

.. automodule:: spectraxgk.validation.benchmarks.cyclone_scan_branches
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.cyclone_scan_seed
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.cyclone_scan_explicit
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.kbm
   :members:
   :no-index:

.. automodule:: spectraxgk.validation.benchmarks.kbm_beta
   :members:

.. automodule:: spectraxgk.validation.benchmarks.kbm_beta_solver_paths
   :members:

.. automodule:: spectraxgk.validation.benchmarks.kbm_linear
   :members:

.. automodule:: spectraxgk.validation.benchmarks.kbm_linear_paths
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.kbm_scan
   :members:

.. automodule:: spectraxgk.validation.benchmarks.etg_linear
   :members:

.. automodule:: spectraxgk.validation.benchmarks.etg_scan
   :members:

.. automodule:: spectraxgk.validation.benchmarks.etg_scan_paths
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.benchmarks.kinetic_linear
   :members:

.. automodule:: spectraxgk.validation.benchmarks.kinetic_scan
   :members:

.. automodule:: spectraxgk.validation.benchmarks.tem
   :members:

.. automodule:: spectraxgk.validation.benchmarks.tem_paths
   :members:
   :private-members:

Validation Gates
----------------

.. automodule:: spectraxgk.validation.gates
   :members:

.. automodule:: spectraxgk.validation.gate_types
   :members:

.. automodule:: spectraxgk.validation.gate_reports
   :members:

Autodiff Validation
-------------------

.. automodule:: spectraxgk.validation.autodiff
   :members:

.. automodule:: spectraxgk.validation.autodiff_finite_difference
   :members:

Parallelization
---------------

.. automodule:: spectraxgk.parallel
   :members:

Parallel Identity Reports
-------------------------

.. automodule:: spectraxgk.parallel.identity
   :members:
   :private-members:

Parallel Batch Mapping
----------------------

.. automodule:: spectraxgk.parallel.batch
   :members:
   :private-members:

Parallel Independent Tasks
--------------------------

.. automodule:: spectraxgk.parallel.independent
   :members:
   :private-members:

Nonlinear Parallel Spectral Core
--------------------------------

.. automodule:: spectraxgk.operators.nonlinear.spectral_core
   :members:
   :private-members:

Nonlinear Domain Decomposition
------------------------------

.. automodule:: spectraxgk.operators.nonlinear.domain_decomposition
   :members:
   :private-members:

Nonlinear Parallel Device-Z Core
--------------------------------

.. automodule:: spectraxgk.operators.nonlinear.device_z
   :members:
   :private-members:

Velocity Sharding Plans
-----------------------

.. automodule:: spectraxgk.parallel.velocity
   :members:
   :no-index:

.. automodule:: spectraxgk.parallel.velocity_plan
   :members:

.. automodule:: spectraxgk.parallel.velocity_hermite
   :members:
   :private-members:

.. automodule:: spectraxgk.parallel.velocity_streaming
   :members:
   :private-members:

.. automodule:: spectraxgk.parallel.velocity_drive
   :members:
   :private-members:

State Sharding Policy
---------------------

.. automodule:: spectraxgk.parallel.state
   :members:

Sharded Integrators
-------------------

.. automodule:: spectraxgk.parallel.integrators
   :members:

Zonal Validation
----------------

.. automodule:: spectraxgk.validation.zonal
   :members:

Zonal Flow Objectives
---------------------

.. automodule:: spectraxgk.objectives.zonal
   :members:

Analysis
--------

.. automodule:: spectraxgk.diagnostics.analysis
   :members:
   :no-index:

Mode Diagnostics
----------------

.. automodule:: spectraxgk.diagnostics.modes
   :members:

Growth-Rate Diagnostics
-----------------------

.. automodule:: spectraxgk.diagnostics.growth_rates
   :members:
   :private-members:

Growth-Rate Fit Kernels
-----------------------

.. automodule:: spectraxgk.diagnostics.growth_fit
   :members:
   :private-members:

Growth-Rate Fit Windows
-----------------------

.. automodule:: spectraxgk.diagnostics.growth_windows
   :members:

Growth-Rate Time Series
-----------------------

.. automodule:: spectraxgk.diagnostics.growth_series
   :members:

Plot Style
----------

.. automodule:: spectraxgk.artifacts.plot_style
   :members:

Runtime Output Plots
--------------------

.. automodule:: spectraxgk.artifacts.runtime_plots
   :members:
   :private-members:

Benchmark And Scan Plots
------------------------

.. automodule:: spectraxgk.artifacts.benchmark_plots
   :members:

Diagnostic Plots
----------------

.. automodule:: spectraxgk.artifacts.diagnostic_plots
   :members:
   :private-members:

Zonal Response Plots
--------------------

.. automodule:: spectraxgk.artifacts.zonal_plots
   :members:

Plotting Facade
---------------

.. automodule:: spectraxgk.artifacts.plotting
   :members:
   :no-index:

Config
------

.. automodule:: spectraxgk.config
   :members:

Normalization
-------------

.. automodule:: spectraxgk.diagnostics.normalization
   :members:

Energy Diagnostics
------------------

.. automodule:: spectraxgk.diagnostics.energy
   :members:
   :private-members:

Transport Diagnostics
---------------------

.. automodule:: spectraxgk.diagnostics.transport
   :members:
   :private-members:

Resolved Diagnostics
--------------------

.. automodule:: spectraxgk.diagnostics.resolved
   :members:
   :private-members:

Runtime Config
--------------

.. automodule:: spectraxgk.workflows.runtime.config
   :members:

Reduced-Model Workflows
-----------------------

.. automodule:: spectraxgk.workflows.reduced_models
   :members:
   :private-members:

Runtime Startup
---------------

.. automodule:: spectraxgk.workflows.runtime.startup
   :members:
   :private-members:

Runtime Policies
----------------

.. automodule:: spectraxgk.workflows.runtime.policies
   :members:
   :private-members:

Runtime Diagnostics
-------------------

.. automodule:: spectraxgk.workflows.runtime.diagnostics
   :members:
   :private-members:

Runtime Diagnostic Arrays
-------------------------

.. automodule:: spectraxgk.workflows.runtime.diagnostic_arrays
   :members:
   :private-members:

Runtime Initial Conditions
--------------------------

.. automodule:: spectraxgk.workflows.runtime.initial_conditions
   :members:
   :private-members:

Runtime Phi Initializer
-----------------------

.. automodule:: spectraxgk.workflows.runtime.initial_phi
   :members:
   :private-members:

Runtime Chunks
--------------

.. automodule:: spectraxgk.workflows.runtime.chunks
   :members:
   :private-members:

Runtime Results
---------------

.. automodule:: spectraxgk.workflows.runtime.results
   :members:
   :private-members:

Runtime Orchestration
---------------------

.. automodule:: spectraxgk.workflows.runtime.orchestration
   :members:
   :private-members:

.. automodule:: spectraxgk.workflows.runtime.orchestration_scan
   :members:
   :private-members:

.. automodule:: spectraxgk.workflows.runtime.orchestration_progress
   :members:
   :private-members:

.. automodule:: spectraxgk.workflows.runtime.orchestration_artifacts
   :members:
   :private-members:

Runtime Commands
----------------

.. automodule:: spectraxgk.workflows.runtime.commands
   :members:
   :private-members:

Runtime TOML Inputs
-------------------

.. automodule:: spectraxgk.workflows.runtime.toml
   :members:
   :private-members:

Runtime Artifact Package
------------------------

.. automodule:: spectraxgk.artifacts
   :members:

Runtime Artifact IO
-------------------

.. automodule:: spectraxgk.artifacts.io
   :members:
   :private-members:

Runtime Restart Artifacts
-------------------------

.. automodule:: spectraxgk.artifacts.restart
   :members:
   :private-members:

Runtime Artifact Linear Writers
-------------------------------

.. automodule:: spectraxgk.artifacts.linear
   :members:
   :private-members:

Runtime Artifact Nonlinear Writers
----------------------------------

.. automodule:: spectraxgk.artifacts.nonlinear
   :members:
   :private-members:

Runtime Artifact Validation
---------------------------

.. automodule:: spectraxgk.artifacts.validation
   :members:
   :private-members:

NetCDF Spectral Layout
----------------------

.. automodule:: spectraxgk.artifacts.spectral_layout
   :members:
   :private-members:

Nonlinear Output NetCDF Geometry
--------------------------------

.. automodule:: spectraxgk.artifacts.nonlinear_netcdf_geometry
   :members:
   :private-members:

Nonlinear Output NetCDF Field Writer
------------------------------------

.. automodule:: spectraxgk.artifacts.nonlinear_netcdf_fields
   :members:
   :private-members:

Nonlinear Output NetCDF Diagnostics Writer
------------------------------------------

.. automodule:: spectraxgk.artifacts.nonlinear_netcdf_diagnostics
   :members:
   :private-members:

Nonlinear Output NetCDF Facade
------------------------------

.. automodule:: spectraxgk.artifacts.nonlinear_netcdf
   :members:
   :private-members:

Runtime Artifact Nonlinear Diagnostics
--------------------------------------

.. automodule:: spectraxgk.artifacts.nonlinear_diagnostics
   :members:
   :private-members:

Quasilinear Transport
---------------------

.. automodule:: spectraxgk.quasilinear
   :members:

Quasilinear Calibration
-----------------------

.. automodule:: spectraxgk.validation.quasilinear.calibration_core
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.quasilinear.calibration_spectrum
   :members:

.. automodule:: spectraxgk.validation.quasilinear.calibration_io
   :members:
   :private-members:

Quasilinear Nonlinear-Window Gates
----------------------------------

.. automodule:: spectraxgk.validation.quasilinear.window_config
   :members:

.. automodule:: spectraxgk.validation.quasilinear.window_statistics
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.quasilinear.window_io
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.quasilinear.window_promotion
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.quasilinear.window_ensemble
   :members:
   :private-members:

Quasilinear Model Selection
---------------------------

.. automodule:: spectraxgk.validation.quasilinear.model_selection
   :members:

Solver Eigen Objectives
-----------------------

.. automodule:: spectraxgk.objectives.eigen
   :members:

Solver Objective Core
---------------------

.. automodule:: spectraxgk.objectives.core
   :members:

Solver Objective Sampling
-------------------------

.. automodule:: spectraxgk.objectives.sampling
   :members:

Solver Geometry Objectives
--------------------------

.. automodule:: spectraxgk.objectives.geometry
   :members:
   :private-members:

Solver Nonlinear-Window Objective
---------------------------------

.. automodule:: spectraxgk.objectives.nonlinear_window
   :members:
   :private-members:

Solver-Ready Gradient Gates
---------------------------

.. automodule:: spectraxgk.objectives.gradient_gates
   :members:
   :private-members:

Solver VMEC/Boozer Gradient Gates
---------------------------------

.. automodule:: spectraxgk.objectives.vmec_boozer_gradients
   :members:
   :private-members:

Solver VMEC-State Helpers
-------------------------

.. automodule:: spectraxgk.objectives.vmec_state
   :members:
   :private-members:

Solver VMEC/Boozer Objectives
-----------------------------

.. automodule:: spectraxgk.objectives.vmec_boozer
   :members:
   :private-members:

Solver VMEC/Boozer Finite-Difference Gates
------------------------------------------

.. automodule:: spectraxgk.objectives.vmec_boozer_fd
   :members:
   :private-members:

Solver VMEC/Boozer Line-Search Gates
------------------------------------

.. automodule:: spectraxgk.objectives.vmec_boozer_line_search
   :members:
   :private-members:

External-VMEC Holdout Planning
------------------------------

.. automodule:: spectraxgk.validation.external_holdout
   :members:

Parallel Decomposition Contracts
--------------------------------

.. automodule:: spectraxgk.parallel.decomposition
   :members:

QA Low-Turbulence Optimization
------------------------------

.. automodule:: spectraxgk.objectives.qa_low_turbulence
   :members:
   :no-index:

QA Low-Turbulence Contracts
---------------------------

.. automodule:: spectraxgk.objectives.qa_low_turbulence_contracts
   :members:

QA Low-Turbulence Reduced Model
-------------------------------

.. automodule:: spectraxgk.objectives.qa_low_turbulence_model
   :members:
   :private-members:

QA Low-Turbulence Residual Gates
--------------------------------

.. automodule:: spectraxgk.objectives.qa_low_turbulence_residuals
   :members:

QA Low-Turbulence Optimizer
---------------------------

.. automodule:: spectraxgk.objectives.qa_low_turbulence_optimizer
   :members:

QA Low-Turbulence Artifacts
---------------------------

.. automodule:: spectraxgk.objectives.qa_low_turbulence_artifacts
   :members:
   :private-members:

VMEC-JAX Transport Objective
----------------------------

.. automodule:: spectraxgk.objectives.vmec_transport
   :members:
   :no-index:

VMEC-JAX Transport Configuration
--------------------------------

.. automodule:: spectraxgk.objectives.vmec_transport_config
   :members:

VMEC-JAX Transport Tables
-------------------------

.. automodule:: spectraxgk.objectives.vmec_transport_tables
   :members:
   :private-members:

VMEC-JAX Transport Branch Gates
-------------------------------

.. automodule:: spectraxgk.objectives.vmec_transport_branch
   :members:

VMEC-JAX Transport Admission
----------------------------

.. automodule:: spectraxgk.validation.stellarator.transport_policies
   :members:

.. automodule:: spectraxgk.validation.stellarator.transport_samples
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.stellarator.transport_landscape
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.stellarator.transport_prelaunch
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.stellarator.transport_campaign
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.stellarator.transport_audit
   :members:
   :private-members:

.. automodule:: spectraxgk.validation.stellarator.transport_selection
   :members:
   :private-members:

VMEC-JAX Transport Gradient
---------------------------

.. automodule:: spectraxgk.objectives.vmec_transport_gradient
   :members:

VMEC-JAX Boundary Chain
-----------------------

.. automodule:: spectraxgk.geometry.vmec_boundary_chain
   :members:

VMEC-JAX Transport Line Search
------------------------------

.. automodule:: spectraxgk.objectives.vmec_transport_line_search
   :members:

VMEC-JAX Candidate Gates
------------------------

.. automodule:: spectraxgk.validation.stellarator.candidate_gate
   :members:

Stellarator ITG Objectives
--------------------------

.. automodule:: spectraxgk.objectives.stellarator
   :members:
   :no-index:

Stellarator ITG Objective Tables
--------------------------------

.. automodule:: spectraxgk.objectives.stellarator_tables
   :members:

Stellarator ITG Residual Gates
------------------------------

.. automodule:: spectraxgk.objectives.stellarator_residuals
   :members:

Stellarator ITG Contracts
-------------------------

.. automodule:: spectraxgk.objectives.stellarator_contracts
   :members:

Stellarator Reduced ITG Model
-----------------------------

.. automodule:: spectraxgk.objectives.stellarator_reduced
   :members:
   :private-members:

Stellarator Objective Portfolio Contracts
-----------------------------------------

.. automodule:: spectraxgk.objectives.portfolio_contracts
   :members:

Stellarator Objective Portfolio Sensitivity
-------------------------------------------

.. automodule:: spectraxgk.objectives.portfolio_sensitivity
   :members:
   :private-members:

Stellarator Objective Portfolio Artifact Guards
-----------------------------------------------

.. automodule:: spectraxgk.objectives.portfolio_artifacts
   :members:
   :private-members:

Runtime Runner
--------------

.. automodule:: spectraxgk.runtime
   :members:
