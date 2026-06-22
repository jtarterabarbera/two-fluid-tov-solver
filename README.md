# Two-Fluid TOV Solver

Repository containing the numerical implementation for the Master Thesis: Interplay of bosonic dark matter with quark matter in compact stars. It provides a modular framework for calculating the Equation of State for various cases and solving the Tolman-Oppenheimer-Volkoff equations to model relativistic stellar structures.

## Repository Structure

* **`core_solver.py`**: The main numerical module. Contains the coupled TOV integrator and routines for unit conversion and boundary extraction.
* **`notebooks/`**: Jupyter Notebooks numbered in execution order.
* **`data/`**: 
  * `eos_library/`: Generated EoS tables.
  * `mr_library/`: Single-fluid Mass-Radius sequences.
  * `mr_library_twofluids/`: Output sequences from the parallelized two-fluid grid scans.

## Execution Pipeline

To reproduce the results of the thesis, the notebooks in the `notebooks/` directory should be run in sequential order:

1. **`01_Generate_EoS_...`**: Generates dimensionless EoS tables for Quark Matter and the different Dark Matter models.
2. **`02_Compute_Mass_Radius_Single_Fluid.ipynb`**: Solves single-fluid TOV equations to obtain M-R relations.
3. **`03_Study_Stability_Single_fluid.ipynb`**: Single-fluid stability analysis via the macroscopic turning-point criterion.
4. **`04_Solve_TOV_Two_Fluids_parallelized.ipynb`**: Parallelized 2D grid scan over central pressures to compute macroscopic properties ($M$, $R$, $N$) of two-fluid admixed stars.
5. **`06_Study_Stability_Two_fluids.ipynb`**: Two-fluid dynamical stability analysis via Jacobian matrix diagonalization.

## Dependencies

Required Python packages:
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`
* `joblib` 
* `tqdm` 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
