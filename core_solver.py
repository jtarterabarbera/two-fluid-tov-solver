import numpy as np
from scipy.integrate import solve_ivp
import pandas as pd
from joblib import Parallel, delayed
import warnings
import gc
from joblib.externals.loky import get_reusable_executor
from tqdm import tqdm
import os


def factors_for_unit_conversion(B_value):
    """
    Computes the factors to convert dimensionless variables into physical units using the scaling '4B' 
    based on a given Bag constant.
    """
    hc = 197.3269804 # MeV fm
    B = B_value ** 4 # MeV^4

    M_p = 1.2209e22  # MeV
    mev_to_kg = 1.78266192e-30 
    msun_kg = 1.98847e30

    # 1. Radius and Mass factors
    factor_R_to_km = (M_p / (2 * np.sqrt(B))) * hc * 1e-18
    factor_M_to_Solar = (M_p **3 / (2 * np.sqrt(B))) * mev_to_kg / msun_kg
    
    # 2. Pressure and Energy Density factors
    factor_P_to_MeV4 = 4 * B
    factor_P_to_MeVfm3 = factor_P_to_MeV4 / hc**3
    
    # 3. Particle Number DENSITY factor (n)
    factor_n_density_to_fm3 = (4 * B) ** (3/4) / hc**3
    
    # 4. TOTAL Particle Number factor (N) 
    factor_N_total = (M_p ** 3) / ((4 * B) ** 0.75)
    
    factors = {
        "factor_R_to_km": factor_R_to_km,
        "factor_M_to_Solar": factor_M_to_Solar,
        "factor_P_to_MeV4": factor_P_to_MeV4,
        "factor_P_to_MeVfm3": factor_P_to_MeVfm3,
        "factor_n_density_to_fm3": factor_n_density_to_fm3,
        "factor_N_total": factor_N_total 
    }
    return factors

def get_initial_conditions(grid_p, grid_e, lower_trim=1, upper_trim=1, percen_samples=0.5):

    """
    Generates an array of central pressures and energy densities from EoS grids. 
    Includes trimming parameters to safely avoid edge-case instabilities at the grid boundaries.
    """

    grid_p = np.asarray(grid_p)
    grid_e = np.asarray(grid_e)

    idx = np.argsort(grid_p)
    grid_p = grid_p[idx]
    grid_e = grid_e[idx]

    mask = (grid_p > 0) & (grid_e > 0) 
    grid_p = grid_p[mask]
    grid_e = grid_e[mask]

    log_p = np.log10(grid_p)
    log_e = np.log10(grid_e)

    pmin, pmax = log_p.min(), log_p.max()
    span = pmax - pmin

    p_low = pmin + span * lower_trim / 100
    p_high = pmax - span * upper_trim / 100

    n = int(len(grid_p) * percen_samples / 100)

    log_p_targets = np.linspace(p_low, p_high, n)
    log_e_targets = np.interp(log_p_targets, log_p, log_e)

    p_init = 10**log_p_targets
    e_init = 10**log_e_targets

    return p_init, e_init

def compute_two_fluid_properties(
        central_e_quark_phys, central_e_dm_phys, 
        grid_e_QM, grid_p_QM, 
        grid_e_DM, grid_p_DM, 
        grid_n_QM, grid_n_DM,
        factors,
        r0=1e-6, 
        save_profiles=False,
        is_smooth=True):
    
    """
    Integrates the coupled two-fluid TOV equations (Quark Matter + Dark Matter).
    Tracks independent fluid boundaries using SciPy events.
    """ 

    # Precompute log-grids to avoid repeated log10 calls inside the ODE
    if grid_e_QM is not None and grid_p_QM is not None:
        log_e_QM = np.log10(grid_e_QM)
        log_p_QM = np.log10(grid_p_QM)
        if grid_n_QM is not None:
            log_n_QM = np.log10(np.maximum(grid_n_QM, 1e-30))
    
    if grid_e_DM is not None and grid_p_DM is not None:
        log_e_DM = np.log10(grid_e_DM)
        log_p_DM = np.log10(grid_p_DM)
        if grid_n_DM is not None:
            log_n_DM = np.log10(np.maximum(grid_n_DM, 1e-30))


    # Setup initial conditions based on central energy densities
    
    # Fluid 1 (Quark Matter)
    if central_e_quark_phys is not None and central_e_quark_phys > 0 and grid_e_QM is not None:
        p_quark_c = 10**np.interp(np.log10(central_e_quark_phys), log_e_QM, log_p_QM)
        m_quark_c = (4/3) * np.pi * r0**3 * central_e_quark_phys
        if grid_n_QM is not None:
            n_quark_c = 10**np.interp(np.log10(central_e_quark_phys), log_e_QM, log_n_QM)
        else:
            n_quark_c = 0.0
        N_quark_c = (4/3) * np.pi * r0**3 * n_quark_c
    else:
        central_e_quark_phys = 0.0
        p_quark_c, m_quark_c, N_quark_c = 0.0, 0.0, 0.0

    # Fluid 2 (Dark Matter)
    if central_e_dm_phys is not None and central_e_dm_phys > 0 and grid_e_DM is not None:
        p_dm_c = 10**np.interp(np.log10(central_e_dm_phys), log_e_DM, log_p_DM)
        m_dm_c = (4/3) * np.pi * r0**3 * central_e_dm_phys
        if grid_n_DM is not None:
            n_dm_c = 10**np.interp(np.log10(central_e_dm_phys), log_e_DM, log_n_DM)
        else:
            n_dm_c = 0.0
        N_dm_c = (4/3) * np.pi * r0**3 * n_dm_c
    else:
        central_e_dm_phys = 0.0
        p_dm_c, m_dm_c, N_dm_c = 0.0, 0.0, 0.0

    x0 = [p_quark_c, m_quark_c, N_quark_c, p_dm_c, m_dm_c, N_dm_c]


    # Define the coupled ODE system
    def ode_TOV(r, x):
        p1, M1, N1, p2, M2, N2 = x
        
        p1_phys = max(0.0, p1)
        p2_phys = max(0.0, p2)

        # Interpolate variables for Fluid 1
        if p1_phys > 0 and grid_e_QM is not None and grid_p_QM is not None:
            e1 = 10**np.interp(np.log10(p1_phys), log_p_QM, log_e_QM)
            if grid_n_QM is not None:
                n1 = 10**np.interp(np.log10(p1_phys), log_p_QM, log_n_QM)
            else:
                n1 = 0.0
        else:
            e1 = 0.0
            n1 = 0.0

        # Interpolate variables for Fluid 2
        if p2_phys > 0 and grid_e_DM is not None and grid_p_DM is not None:
            e2 = 10**np.interp(np.log10(p2_phys), log_p_DM, log_e_DM)
            if grid_n_DM is not None:
                n2 = 10**np.interp(np.log10(p2_phys), log_p_DM, log_n_DM)
            else:
                n2 = 0.0
        else:
            e2 = 0.0  
            n2 = 0.0  
            
        M = M1 + M2

        if r < 1e-9:
            return [0.0, 4 * np.pi * r**2 * e1, 0.0,  0.0, 4 * np.pi * r**2 * e2, 0.0]

        term1_q = (e1 + p1_phys)
        term1_dm = (e2 + p2_phys)
        
        # Handle singularity limits near the origin
        if M < 1e-30:
            term2 = 1.0
        else:
            term2 = 1 + 4 * np.pi * r**3 * (p1_phys + p2_phys) / M
            
        term3 = 1 - 2 * M / r

        if term3 <= 1e-9:
            return [0, 0, 0, 0, 0, 0]

        n_particles_1 = 4 * np.pi * r**2 * n1 / np.sqrt(term3)
        n_particles_2 = 4 * np.pi * r**2 * n2 / np.sqrt(term3)

        factor = - (M / r**2) / term3

        # Derivatives
        if p1 <= 0:
            dp1dr, dM1dr, dN1dr = 0.0, 0.0, 0.0
        else:
            dp1dr = factor * term1_q * term2
            dM1dr = 4 * np.pi * r**2 * e1
            dN1dr = n_particles_1

        if p2 <= 0:
            dp2dr, dM2dr, dN2dr = 0.0, 0.0, 0.0
        else:
            dp2dr = factor * term1_dm * term2
            dM2dr = 4 * np.pi * r**2 * e2
            dN2dr = n_particles_2

        return [dp1dr, dM1dr, dN1dr, dp2dr, dM2dr, dN2dr]


    # Boundary tracking events
    def event_DM_surface(r, x):
        return x[3] 
    event_DM_surface.terminal = False
    event_DM_surface.direction = -1

    def event_Quark_surface(r, x):
        return x[0]
    event_Quark_surface.terminal = False
    event_Quark_surface.direction = -1

    def event_Stop_Integration(r, x):
        return max(x[0], x[3]) 
    event_Stop_Integration.terminal = True
    event_Stop_Integration.direction = -1


    # Integration execution
    if is_smooth:
        sol = solve_ivp(
            ode_TOV, 
            (r0, 1e10), 
            x0, 
            events=[event_DM_surface, event_Quark_surface, event_Stop_Integration],
            rtol=1e-8, 
            atol=1e-10,
            first_step=1e-7     
        )
    else:
        sol = solve_ivp(
            ode_TOV, 
            (r0, 1e10), 
            x0, 
            events=[event_DM_surface, event_Quark_surface, event_Stop_Integration],
            rtol=1e-8, 
            atol=1e-10,
            first_step=1e-7,
            method='Radau',
        )


    # Surface extraction
    r_adim = sol.t
    p1_adim, m1_adim, n1_adim, p2_adim, m2_adim, n2_adim = sol.y 

    # Total macroscopic radius and mass
    if len(sol.t_events[2]) > 0:
        R_total_adim = sol.t_events[2][0]
        final_state = sol.y_events[2][0]
        M_total_adim = final_state[1] + final_state[4] 
        N_total_adim = final_state[2] + final_state[5]
    else:
        R_total_adim = r_adim[-1]
        M_total_adim = m1_adim[-1] + m2_adim[-1]
        N_total_adim = n1_adim[-1] + n2_adim[-1]


    # Fluid 1
    if p_quark_c > 0:
        if len(sol.t_events[1]) > 0:
            R_quark_adim = sol.t_events[1][0]
            M_quark_adim = sol.y_events[1][0][1]
            N_quark_adim = sol.y_events[1][0][2]
        else:
            if len(sol.t_events[2]) > 0:
                R_quark_adim = sol.t_events[2][0]
                M_quark_adim = sol.y_events[2][0][1]
                N_quark_adim = sol.y_events[2][0][2]
            else:
                R_quark_adim = r_adim[-1]
                M_quark_adim = m1_adim[-1]
                N_quark_adim = n1_adim[-1]
    else:
        R_quark_adim, M_quark_adim, N_quark_adim = 0.0, 0.0, 0.0

    # Fluid 2
    if p_dm_c > 0:
        if len(sol.t_events[0]) > 0:
            R_dm_adim = sol.t_events[0][0]
            M_dm_adim = sol.y_events[0][0][4]
            N_dm_adim = sol.y_events[0][0][5]
        else:
            if len(sol.t_events[2]) > 0:
                R_dm_adim = sol.t_events[2][0]
                M_dm_adim = sol.y_events[2][0][4]
                N_dm_adim = sol.y_events[2][0][5]
            else:
                R_dm_adim = r_adim[-1]
                M_dm_adim = m2_adim[-1]
                N_dm_adim = n2_adim[-1]
    else:
        R_dm_adim, M_dm_adim, N_dm_adim = 0.0, 0.0, 0.0

    R_total_km = R_total_adim * factors["factor_R_to_km"]
    M_total_sol = M_total_adim * factors["factor_M_to_Solar"]
      
    # Format properties dictionary
    results = {
        "e_c_qm": central_e_quark_phys * factors["factor_P_to_MeVfm3"],
        "e_c_dm": central_e_dm_phys * factors["factor_P_to_MeVfm3"],
        "p_c_qm": p_quark_c * factors["factor_P_to_MeVfm3"],
        "p_c_dm": p_dm_c * factors["factor_P_to_MeVfm3"],

        "R_quark_km": R_quark_adim * factors["factor_R_to_km"],
        "R_dm_km": R_dm_adim * factors["factor_R_to_km"],
        "R_total_km": R_total_km,

        "M_quark_sol": M_quark_adim * factors["factor_M_to_Solar"],
        "M_dm_sol": M_dm_adim * factors["factor_M_to_Solar"],
        "M_total_sol": M_total_sol,
        
        "N_quark": N_quark_adim * factors["factor_N_total"],
        "N_dm": N_dm_adim * factors["factor_N_total"],
        "N_total": (N_total_adim) * factors["factor_N_total"]
    }

    if save_profiles:
        results["R_profile_km"] = r_adim * factors["factor_R_to_km"]
        
        results["M_quark_profile"] = m1_adim * factors["factor_M_to_Solar"]
        results["M_dm_profile"] = m2_adim * factors["factor_M_to_Solar"]
        results["M_total_profile"] = results["M_quark_profile"] + results["M_dm_profile"]
        
        p_q_prof = np.maximum(p1_adim * factors["factor_P_to_MeVfm3"], 0.0)
        p_dm_prof = np.maximum(p2_adim * factors["factor_P_to_MeVfm3"], 0.0)
        results["P_quark_profile"] = p_q_prof
        results["P_dm_profile"] = p_dm_prof
        results["P_total_profile"] = p_q_prof + p_dm_prof
        
        N_q_prof = np.maximum(n1_adim * factors["factor_N_total"], 0.0)
        N_dm_prof = np.maximum(n2_adim * factors["factor_N_total"], 0.0)
        results["N_quark_profile"] = N_q_prof
        results["N_dm_profile"] = N_dm_prof
        results["N_total_profile"] = N_q_prof + N_dm_prof

    return results

def process_single_combination(
    p_c_OM, e_c_OM, p_c_DM, e_c_DM, 
    grid_e_QM, grid_p_QM, grid_n_QM, 
    grid_e_DM, grid_p_DM, grid_n_DM, 
    factors, is_smooth=True
):
    r0 = min(1e-6, 1e-3 / np.sqrt(e_c_OM))

    results = compute_two_fluid_properties(
        e_c_OM, e_c_DM, 
        grid_e_QM, grid_p_QM, 
        grid_e_DM, grid_p_DM, 
        grid_n_QM, grid_n_DM,
        factors,
        r0=r0,
        save_profiles=False,
        is_smooth=is_smooth
    )
    
    return results

def run_generic_parallel_scan(
    config_name, eos_om_path, eos_dm_path, B_val_MeV,
    dm_p_min=1, dm_p_max=1e4, dm_steps=200,
    qm_lower_trim=50, qm_upper_trim=20, qm_samples=0.2,
    chunk_size=10000, smoothness=True, njobs=-1
):
    """
    Computes a 2D grid of two-fluid TOV solutions in parallel chunks 
    and appends the results iteratively to a CSV file.
    """
    factors = factors_for_unit_conversion(B_value=B_val_MeV)
    B = B_val_MeV**4 
    hc = 197.3269804 # MeV fm

    # Load EoS tables
    df_qm = pd.read_csv(eos_om_path)
    grid_e_QM, grid_p_QM, grid_n_QM = df_qm['Energy Density'].values, df_qm['Pressure'].values, df_qm['Number Density'].values

    df_dm = pd.read_csv(eos_dm_path)
    grid_e_DM, grid_p_DM, grid_n_DM = df_dm['Energy Density'].values, df_dm['Pressure'].values, df_dm['Number Density'].values

    # Set up DM central pressure grid (dimensionless)
    p_c_dm_phys = np.logspace(np.log10(dm_p_min), np.log10(dm_p_max), dm_steps)
    p_c_dm_adim = p_c_dm_phys * hc**3 / (4 * B)
    e_c_dm_adim = np.interp(p_c_dm_adim, grid_p_DM, grid_e_DM)

    # Set up QM central conditions
    p_init_qm, e_init_qm = get_initial_conditions(
        grid_p_QM, grid_e_QM, 
        lower_trim=qm_lower_trim, upper_trim=qm_upper_trim, percen_samples=qm_samples
    )

    # Generate the 2D parameter space combinations
    tasks = [
        (p_qm, e_qm, p_dm, e_dm)
        for p_qm, e_qm in zip(p_init_qm, e_init_qm)
        for p_dm, e_dm in zip(p_c_dm_adim, e_c_dm_adim)
    ]

    print(f"[{config_name}] Grid size: {len(tasks)} combinations (QM: {len(p_init_qm)}, DM: {dm_steps})")
    
    csv_filename = f'../data/mr_library_twofluids/mr_{config_name}.csv'

    # Execute in chunks to prevent joblib memory leaks on massive arrays
    for i in range(0, len(tasks), chunk_size):
        chunk = tasks[i : i + chunk_size]
        print(f"Processing chunk {i // chunk_size + 1}/{(len(tasks) - 1) // chunk_size + 1}...")
        
        jobs = (
            delayed(process_single_combination)(
                p_qm, e_qm, p_dm, e_dm,
                grid_e_QM, grid_p_QM, grid_n_QM,
                grid_e_DM, grid_p_DM, grid_n_DM,
                factors, is_smooth=smoothness
            ) 
            for p_qm, e_qm, p_dm, e_dm in chunk
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results = list(tqdm(
                Parallel(n_jobs=njobs, return_as="generator")(jobs), 
                total=len(chunk), 
                leave=False
            ))
            
        df_chunk = pd.DataFrame(results)
        
        # Append to CSV dynamically
        write_mode, write_header = ('a', False) if os.path.isfile(csv_filename) else ('w', True)
        df_chunk.to_csv(csv_filename, mode=write_mode, header=write_header, index=False)
            
        # Cleanup
        del results, df_chunk, jobs
        gc.collect() 
        get_reusable_executor().shutdown(wait=True)

    print(f"[{config_name}] Done. Results saved to {csv_filename}\n")