# Excel File Review: COIN_equality_v1.xlsm

## Critical Issues Found

### 1. **Ecum_t is Empty (CRITICAL BUG)**
- **Location:** Step1_Test sheet, Row 6, Column C
- **Current Value:** Empty/None
- **Expected Value:** 2494300000000 (from Ecum_initial)
- **Impact:** This causes all cumulative emissions calculations to be wrong
- **Fix:** Set cell C6 to the value from Config!Ecum_initial (row 19)

### 2. **Ecum_next is Wrong (CRITICAL BUG)**
- **Location:** Step1_Test sheet, Row 38, Column C
- **Current Value:** 31015233857.754307
- **Expected Value:** 2525315233857.7544
- **Formula:** `=MAX(0, Ecum_t + dt*E)`
- **Issue:** Because Ecum_t is empty, this only calculates `dt*E` instead of `Ecum_t + dt*E`
- **Impact:** Cumulative emissions tracking is completely broken
- **Fix:** First fix issue #1, then this formula will work correctly

## Configuration Comparison

The Excel Config sheet was compared against `config_test_DICE_2x_0.02_10k_0.67.json`. All values match correctly:

| Parameter | Excel Value | JSON Value | Status |
|-----------|-------------|------------|--------|
| alpha | 0.3 | 0.3 | ✓ Match |
| delta | 0.1 | 0.1 | ✓ Match |
| s | 0.28 | N/A (from optimization) | ✓ OK |
| psi2 | 0.006934 | 0.006934 | ✓ Match |
| y_damage_halfsat | 10000 | 10000 | ✓ Match |
| Gini_initial | 0.67 | 0.67 | ✓ Match |
| theta1 | 695.177385 | 695.177385 | ✓ Match |
| sigma | 0.000291355 | 0.000291355 | ✓ Match |
| Ecum_initial | 2494300000000 | 2494300000000 | ✓ Match |
| K_initial | 295000000000000 | 295000000000000 | ✓ Match |

**Note:** The Excel uses the "2x" scenario with doubled psi2 (0.006934 vs 0.003467) and higher Gini (0.67 vs 0.0), which matches the `config_test_DICE_2x_0.02_10k_0.67.json` configuration.

## Formula Verification

### Formulas That Look Correct

| Variable | Excel Formula | Python Equivalent | Status |
|----------|--------------|-------------------|--------|
| Y_gross | `=A * K_t^alpha * L^(1-alpha)` | `A * (K ** alpha) * (L ** (1 - alpha))` | ✓ Match |
| Y_damaged | `=Y_gross*(1-Omega)` | `(1 - Omega) * Y_gross` | ✓ Match |
| Y_net | `=Y_damaged*(1-Lambda)` | `(1 - Lambda) * Y_damaged` | ✓ Match |
| E | `=sigma * (1 - mu) * Y_gross` | `sigma * (1 - mu) * Y_gross` | ✓ Match |
| dK/dt | `=s*Y_net - delta*K_t` | `s * Y_net - delta * K` | ✓ Match |
| K_next | `=K_t + dt*dKdt` | Not directly in code (handled by integrator) | ✓ OK |
| Ecum_next | `=MAX(0, Ecum_t + dt*E)` | Not directly in code (handled by integrator) | ✓ Formula OK, but input data is wrong |

### Variables Needing Manual Verification

The following complex variables involve iterative calculations and should be verified:

1. **Omega (Ω)** - Aggregate climate damage fraction
   - Depends on income-dependent damage calculation
   - May involve hypergeometric functions (see HYP2F1_Module.bas)
   - Python: `calculate_climate_damage_and_gini_effect()`

2. **mu (μ)** - Abatement fraction
   - Formula: `mu = min(mu_max, (abatecost * theta2 / (Epot * theta1)) ** (1 / theta2))`
   - Needs verification against Excel calculation

3. **Lambda (Λ)** - Abatement cost fraction
   - Formula: `Lambda = abatecost / Y_damaged`
   - Should be straightforward

4. **U (Utility)** - CRRA utility with Gini adjustment
   - Complex formula involving eta and G_eff
   - Python has special case for eta ≈ 1.0

## Missing Variables in Excel

The following variables from the Python code don't appear to have clear equivalents in the Excel file:

1. **Gini_climate** - Post-climate-damage Gini (output of damage calculation)
2. **dGini_dt** - Gini restoration dynamics
3. **Gini_step_change** - Instantaneous Gini step change

## Recommendations

### Immediate Fixes Required

1. **Set Ecum_t value** in cell C6 to reference Config!B19 (Ecum_initial)
   - This is the most critical fix

2. **Verify the Ecum_next calculation** after fixing #1
   - Should equal: 2525315233857.7544 for the first time step

### Verification Checklist

Before trusting the Excel calculations:

- [ ] Fix Ecum_t (cell C6)
- [ ] Verify Ecum_next updates correctly
- [ ] Compare Omega calculation with Python code
- [ ] Compare mu calculation with Python code
- [ ] Compare U (utility) with Python code
- [ ] Test with multiple values of f to ensure formulas are robust
- [ ] Verify that iterative calculation is enabled (1000 iterations, 1E-6 tolerance)
- [ ] Check if HYP2F1_Module.bas is properly loaded for hypergeometric functions

### Additional Notes

1. The Notes sheet mentions that cells sourced from JSON are highlighted in light yellow - verify this visual marking is correct

2. The damage_mode switch ('uniform' vs 'income') should be tested to ensure both modes work correctly

3. Consider adding data validation to prevent Ecum_t from being accidentally cleared

4. Consider adding a "First Time Step" checkbox that automatically populates:
   - K_t = K_initial
   - Ecum_t = Ecum_initial
   - Gini_t = Gini_initial (if applicable)
