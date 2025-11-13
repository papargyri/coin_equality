"""
Simple verification script: Run scenarios with different constant f values.

This tests the forward model (integrate_model) with various abatement fractions
to verify calculations match Excel expectations.
"""

import sys
from parameters import load_configuration, create_constant_control_from_scalar, ModelConfiguration
from economic_model import integrate_model
import json

def run_scenario(base_config, f_value, scenario_name):
    """Run integration with a constant f value."""

    # Create a constant control function with the specified f
    # The savings rate s is taken from the base config
    _, s_value = base_config.control_function(0.0)  # Get s from base config
    constant_control = create_constant_control_from_scalar(f_value, s_value)

    # Create new config with this control function
    config = ModelConfiguration(
        run_name=f"{base_config.run_name}_f{f_value:.2f}",
        scalar_params=base_config.scalar_params,
        time_functions=base_config.time_functions,
        integration_params=base_config.integration_params,
        optimization_params=base_config.optimization_params,
        initial_state=base_config.initial_state,
        control_function=constant_control
    )

    # Run integration (only store detailed output for t=0)
    results = integrate_model(config, store_detailed_output=True)

    return results

def main():
    # Load base configuration
    config_file = 'config_test_DICE.json'
    print("=" * 80)
    print("VERIFICATION: Python Solver vs Excel Model")
    print("=" * 80)
    print(f"\nLoading base configuration: {config_file}")

    base_config = load_configuration(config_file)

    print(f"\nBase Configuration:")
    print(f"  Run name: {base_config.run_name}")
    print(f"  Time span: {base_config.integration_params.t_start} to {base_config.integration_params.t_end} yr")
    print(f"  alpha (capital share): {base_config.scalar_params.alpha}")
    print(f"  delta (depreciation): {base_config.scalar_params.delta}")
    print(f"  eta (risk aversion): {base_config.scalar_params.eta}")
    print(f"  fract_gdp: {base_config.scalar_params.fract_gdp}")
    print(f"  Initial K: {base_config.scalar_params.K_initial:.6e}")
    print(f"  Initial Ecum: {base_config.scalar_params.Ecum_initial:.6e}")
    print(f"  Initial Gini: {base_config.scalar_params.Gini_initial}")

    # Test scenarios
    scenarios = [
        {'name': 'No Abatement (all to redistribution)', 'f': 0.0},
        {'name': 'Low Abatement (10%)', 'f': 0.1},
        {'name': 'Medium Abatement (50%)', 'f': 0.5},
        {'name': 'Full Abatement (all to abatement)', 'f': 1.0},
    ]

    print("\n" + "=" * 80)
    print("RUNNING SCENARIOS (First Time Step Only)")
    print("=" * 80)

    all_results = []

    for scenario in scenarios:
        print(f"\n{'─' * 80}")
        print(f"Scenario: {scenario['name']} (f = {scenario['f']:.3f})")
        print(f"{'─' * 80}")

        results = run_scenario(base_config, scenario['f'], scenario['name'])

        # Extract t=0 values
        idx = 0
        result_summary = {
            'scenario': scenario['name'],
            'f': scenario['f'],
            't': results['t'][idx],
            'K': results['K'][idx],
            'Ecum': results['Ecum'][idx],
            'Gini': results['Gini'][idx],
            'L': results['L'][idx],
            'Y_gross': results['Y_gross'][idx],
            'delta_T': results['delta_T'][idx],
            'Omega': results['Omega'][idx],
            'Y_damaged': results['Y_damaged'][idx],
            'y': results['y'][idx],
            'delta_c': results['delta_c'][idx],
            'abatecost': results['abatecost'][idx],
            'mu': results['mu'][idx],
            'Lambda': results['Lambda'][idx],
            'Y_net': results['Y_net'][idx],
            'y_eff': results['y_eff'][idx],
            'G_eff': results['G_eff'][idx],
            'U': results['U'][idx],
            'E': results['E'][idx],
        }

        all_results.append(result_summary)

        print(f"\nState Variables (t={result_summary['t']:.1f}):")
        print(f"  K (capital):        {result_summary['K']:.6e} $")
        print(f"  Ecum (emissions):   {result_summary['Ecum']:.6e} tCO2")
        print(f"  Gini:               {result_summary['Gini']:.6f}")
        print(f"  L (population):     {result_summary['L']:.6e}")

        print(f"\nProduction:")
        print(f"  Y_gross:            {result_summary['Y_gross']:.6e} $/yr")
        print(f"  ΔT:                 {result_summary['delta_T']:.6f} °C")
        print(f"  Ω (damage):         {result_summary['Omega']:.6f}")
        print(f"  Y_damaged:          {result_summary['Y_damaged']:.6e} $/yr")

        print(f"\nIncome & Redistribution:")
        print(f"  y (per-capita):     {result_summary['y']:.2f} $/person/yr")
        print(f"  Δc (redistributed): {result_summary['delta_c']:.2f} $/person/yr")

        print(f"\nAbatement:")
        print(f"  Abatement cost:     {result_summary['abatecost']:.6e} $/yr")
        print(f"  μ (abatement frac): {result_summary['mu']:.6f}")
        print(f"  Λ (cost fraction):  {result_summary['Lambda']:.6f}")
        print(f"  E (emissions):      {result_summary['E']:.6e} tCO2/yr")

        print(f"\nFinal Outcome:")
        print(f"  Y_net:              {result_summary['Y_net']:.6e} $/yr")
        print(f"  y_eff:              {result_summary['y_eff']:.2f} $/person/yr")
        print(f"  G_eff:              {result_summary['G_eff']:.6f}")
        print(f"  U (mean utility):   {result_summary['U']:.6f}")

    # Summary comparison table
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print("\n{:<40} {:>8} {:>12} {:>12} {:>12}".format("Scenario", "f", "μ", "U", "y_eff"))
    print("─" * 80)
    for r in all_results:
        print("{:<40} {:>8.3f} {:>12.6f} {:>12.6f} {:>12.2f}".format(
            r['scenario'][:40], r['f'], r['mu'], r['U'], r['y_eff']
        ))

    print("\n" + "=" * 80)
    print("EXCEL COMPARISON INSTRUCTIONS")
    print("=" * 80)
    print("\nTo verify these results in Excel:")
    print("1. Open: 1_time_step_calc/COIN_equality_v1.xlsx")
    print("2. Go to 'Step1_Test' sheet")
    print("3. Set cell C7 (f) to each test value: 0.0, 0.1, 0.5, 1.0")
    print("4. Press F9 to recalculate after each change")
    print("5. Compare key cells:")
    print("   - C28: μ (abatement fraction)")
    print("   - C31: y_eff (effective per-capita income)")
    print("   - C33: G_eff (effective Gini)")
    print("   - C34: U (mean utility)")
    print("\nNote: Excel Config sheet should match Python config:")
    print(f"  - alpha = {base_config.scalar_params.alpha}")
    print(f"  - delta = {base_config.scalar_params.delta}")
    print(f"  - eta = {base_config.scalar_params.eta}")
    print(f"  - s = 0.28 (from optimization_parameters.initial_guess_s)")
    print(f"  - fract_gdp = {base_config.scalar_params.fract_gdp}")

    print("\n" + "=" * 80)

if __name__ == '__main__':
    main()
