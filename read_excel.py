import openpyxl
import json

# Load the Excel file
wb = openpyxl.load_workbook('/Users/lamprinipapargyri/Code/coin_equality/1_time_step_calc/COIN_equality_v1.xlsx', data_only=True)

# Read Config sheet
print("="*80)
print("CONFIG SHEET")
print("="*80)
ws_config = wb['Config']
for row in range(1, 30):
    cell_b = ws_config[f'B{row}'].value
    cell_c = ws_config[f'C{row}'].value
    if cell_b is not None:
        print(f'{cell_b}: {cell_c}')

# Read Step1_Test sheet inputs and outputs
print("\n" + "="*80)
print("STEP1_TEST SHEET - ONE TIME STEP CALCULATION")
print("="*80)
ws_test = wb['Step1_Test']

data = {
    'inputs': {},
    'intermediates': {},
    'outputs': {}
}

# Key input cells
data['inputs']['f'] = ws_test['C7'].value
data['inputs']['A'] = ws_test['C8'].value
data['inputs']['L'] = ws_test['C11'].value
data['inputs']['K'] = ws_test['C12'].value
data['inputs']['alpha'] = ws_test['C13'].value
data['inputs']['s'] = ws_test['C14'].value

# Intermediate calculations
data['intermediates']['y'] = ws_test['C15'].value
data['intermediates']['f_gdp'] = ws_test['C16'].value
data['intermediates']['Y_gross'] = ws_test['C20'].value
data['intermediates']['Omega'] = ws_test['C21'].value
data['intermediates']['Y_damaged'] = ws_test['C22'].value
data['intermediates']['c_redist'] = ws_test['C24'].value
data['intermediates']['AbateCost'] = ws_test['C25'].value
data['intermediates']['theta1'] = ws_test['C26'].value
data['intermediates']['theta2'] = ws_test['C27'].value
data['intermediates']['mu'] = ws_test['C28'].value
data['intermediates']['Lambda'] = ws_test['C29'].value
data['intermediates']['Y_net'] = ws_test['C30'].value
data['intermediates']['y_eff'] = ws_test['C31'].value

# Output
data['outputs']['G_eff'] = ws_test['C33'].value
data['outputs']['U'] = ws_test['C34'].value

print(json.dumps(data, indent=2))
