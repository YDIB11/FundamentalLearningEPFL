import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

capture = False
for cell in nb['cells']:
    mid = cell.get('metadata', {}).get('id')
    if mid == 'E6fOj_gVl0Xl':
        capture = True
        continue
    if mid == 'vV1MU1qZba_h':
        break
    if capture:
        print('---', cell.get('cell_type'), cell.get('metadata', {}).get('id'))
        for line in cell.get('source', []):
            print(line.rstrip())
        print()
