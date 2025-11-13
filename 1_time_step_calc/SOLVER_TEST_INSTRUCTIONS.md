# How to Verify Excel Solver is Working

## Current Status
✓ Excel file is calculating correctly
✓ All formulas are working
✓ Ecum_t is fixed
✓ Current f = 0.5, U = 14.364528

## Test: Did Solver Actually Run?

### Quick Manual Test

1. **Open COIN_equality_v1.xlsm in Excel**

2. **Record current values**:
   - Cell C7 (f) = 0.5
   - Cell C34 (U) = 14.364528

3. **Change f manually** to test if utility changes:
   - Set C7 = 0.2
   - Watch C34 change (U should decrease)
   - Set C7 = 0.8
   - Watch C34 change (U should decrease again)
   - Set C7 back to 0.5

4. **This proves formulas are working!**

## How to Run Solver (Optimization)

### Step-by-Step Solver Setup:

1. **Open Solver**:
   - Mac: Data → Solver (may need to enable in Tools → Add-ins)
   - Windows: Data tab → Solver button

2. **Set Solver Parameters**:
   ```
   Set Objective: $C$34  (the U cell)
   To: Max
   By Changing Variable Cells: $C$7  (the f cell)

   Subject to Constraints:
   Add: $C$7 >= 0
   Add: $C$7 <= 1
   Add: $C$31 > 0  (y_eff must be positive)
   ```

3. **Solver Options** (click Options button):
   ```
   Solving Method: GRG Nonlinear
   Max Time: 100 seconds
   Iterations: 100
   Precision: 0.000001
   ```

4. **Click SOLVE**

5. **Check Results**:
   - Solver should say "Solver found a solution"
   - Click "Keep Solver Solution"
   - Note the optimized f value
   - Note the maximized U value

## Expected Results

Based on the model structure, the optimal f should be somewhere between 0.3 and 0.7.

### What to Look For:

**If Solver worked**:
- f will change to an optimized value (probably not exactly 0.5)
- U will reach a maximum
- Solver dialog will say "Solver found a solution. All constraints and optimality conditions are satisfied."

**If Solver failed**:
- Dialog will say "Solver could not find a feasible solution"
- Or "The objective cell values do not converge"
- Check that:
  - Iterative calculation is enabled (File → Options → Formulas → Enable iterative calculation)
  - Max iterations = 1000, Max change = 1E-6
  - All formulas are calculating without errors

## Test Different Damage Modes

The Excel has a switch in cell C8 for damage_mode ('uniform' vs 'income').

Try both:
1. Set C8 = "uniform" → run Solver → note optimal f
2. Set C8 = "income" → run Solver → note optimal f
3. Compare: income-dependent damage should give different optimal f

## Troubleshooting

### If U shows #VALUE! or #NUM! errors:
- Check that HYP2F1_Module.bas is loaded (for hypergeometric functions)
- Check that iterative calculation is enabled
- Try setting f = 0.5 manually first to see if formulas calculate

### If Solver doesn't find a solution:
- Try starting from different initial f values (0.3, 0.5, 0.7)
- Check constraint C31 > 0 (y_eff positive)
- Simplify by removing some constraints and add back one at a time

### If Solver gives unreasonable results (f = 0 or f = 1):
- Check that fract_gdp makes sense (should be small, like 0.02)
- Verify damage parameters aren't too extreme
- Check that theta2 > 1 (should be 2.6)

## Comparison with Python

To compare with Python optimization results, run:
```bash
cd /Users/lamprinipapargyri/Code/coin_equality
python test_optimization.py config_test_DICE_2x_0.02_10k_0.67.json
```

The Python code will find optimal f(t) trajectory. For a single time step:
- Compare your Excel optimal f with Python's f(0)
- They should be close (within 0.05)
- Small differences are okay due to different optimization algorithms
