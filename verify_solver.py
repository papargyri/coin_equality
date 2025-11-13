"""
Verification script to compare Python solver with Excel model.

This script runs multiple scenarios and compares Python calculations
with expected Excel results for validation.
"""

import numpy as np
from economic_model import calculate_tendencies, integrate_model
from parameters import ModelConfiguration, ScalarParams, IntegrationParams, OptimizationParams, create_constant_control_from_scalar
import json

print("=" * 80)
print("COIN EQUALITY: PYTHON SOLVER VERIFICATION")
print("=" * 80)

# Read configuration parameters (matching Excel Config sheet)
config_params = {
    'alpha': 0.3,
    'delta': 0.1,
    's': 0.28,
    'rho': 0.01,
    'eta': 0.95,
    'fract_gdp': 0.02,
    'mu_max': 1.2,
    'theta2': 2.6,
    'theta1_initial': 695.177385,
    'sigma_initial': 0.000291355,
    'k_climate': 5e-13,
    'psi1': 0,
    'psi2': 0.006934,
    'y_damage_halfsat': 10000,
    'Gini_initial': 0.67,
    'Gini_fract': 0,
    'Gini_restore': 0,
    'Ecum_initial': 2494300000000.0,
    'K_initial': 295000000000000.0,
    'A_initial': 739.619,
    'L_initial': 7752000000.0,
}

print("\nConfiguration Parameters:")
print(json.dumps(config_params, indent=2))

# Test scenarios with different f values
test_scenarios = [
    {'name': 'f=0.0 (all redistribution)', 'f': 0.0},
    {'name': 'f=0.1 (10% to abatement)', 'f': 0.1},
    {'name': 'f=0.5 (50% to abatement)', 'f': 0.5},
    {'name': 'f=1.0 (all to abatement)', 'f': 1.0},
]

# Function to create constant time functions
def create_constant_function(value):
    return lambda t: value

print("\n" + "=" * 80)
print("SINGLE TIME STEP VERIFICATION (t=0)")
print("=" * 80)

# For each scenario, calculate one timestep and show results
for scenario in test_scenarios:
    print(f"\n{'─' * 80}")
    print(f"SCENARIO: {scenario['name']}")
    print(f"{'─' * 80}")

    f_value = scenario['f']

    # Create state at t=0
    state = {
        'K': config_params['K_initial'],
        'Ecum': config_params['Ecum_initial'],
        'Gini': config_params['Gini_initial'],
    }

    # Create params for t=0
    params = {
        'alpha': config_params['alpha'],
        'delta': config_params['delta'],
        's': config_params['s'],
        'k_climate': config_params['k_climate'],
        'eta': config_params['eta'],
        'rho': config_params['rho'],
        't': 0.0,
        'A': config_params['A_initial'],
        'L': config_params['L_initial'],
        'sigma': config_params['sigma_initial'],
        'theta1': config_params['theta1_initial'],
        'theta2': config_params['theta2'],
        'mu_max': config_params['mu_max'],
        'fract_gdp': config_params['fract_gdp'],
        'Gini_initial': config_params['Gini_initial'],
        'Gini_fract': config_params['Gini_fract'],
        'Gini_restore': config_params['Gini_restore'],
        'f': f_value,
        'psi1': config_params['psi1'],
        'psi2': config_params['psi2'],
        'y_damage_halfsat': config_params['y_damage_halfsat'],
    }

    # Calculate tendencies (one time step)
    results = calculate_tendencies(state, params, store_detailed_output=True)

    # Display key results
    print(f"\nInputs:")
    print(f"  f (abatement fraction): {f_value:.3f}")
    print(f"  K (capital): {state['K']:.6e}")
    print(f"  Ecum (cumulative emissions): {state['Ecum']:.6e}")
    print(f"  Gini: {state['Gini']:.3f}")
    print(f"  A (TFP): {params['A']:.3f}")
    print(f"  L (population): {params['L']:.6e}")

    print(f"\nKey Outputs:")
    print(f"  Y_gross (gross production): ${results['Y_gross']:.6e}")
    print(f"  delta_T (temperature change): {results['delta_T']:.6f} °C")
    print(f"  Omega (climate damage frac): {results['Omega']:.6f}")
    print(f"  Y_damaged (after damage): ${results['Y_damaged']:.6e}")
    print(f"  y (per-capita income): ${results['y']:.2f}")
    print(f"  delta_c (redistributable): ${results['delta_c']:.2f}")
    print(f"  abatecost (total): ${results['abatecost']:.6e}")
    print(f"  mu (abatement fraction): {results['mu']:.6f}")
    print(f"  Lambda (abatement cost frac): {results['Lambda']:.6f}")
    print(f"  Y_net (after abatement): ${results['Y_net']:.6e}")
    print(f"  y_eff (effective income): ${results['y_eff']:.2f}")
    print(f"  G_eff (effective Gini): {results['G_eff']:.6f}")
    print(f"  U (mean utility): {results['U']:.6f}")
    print(f"  E (emissions): {results['E']:.6e} tCO2/yr")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
print("\nTo compare with Excel:")
print("1. Open: 1_time_step_calc/COIN_equality_v1.xlsx")
print("2. Set cell C7 (f) to each test value (0.0, 0.1, 0.5, 1.0)")
print("3. Press F9 to recalculate")
print("4. Compare cells C34 (U), C28 (mu), C30 (Y_net), C31 (y_eff), C33 (G_eff)")
