import json
path = r"TP6/regularized_linear_regression.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'k3ditaL76wm8':
        break
else:
    raise RuntimeError('Q5 markdown not found')
