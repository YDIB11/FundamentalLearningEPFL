import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb['cells']:
    meta = cell.get('metadata', {})
    if meta.get('id') == 'k3ditaL76wm8':
        for line in cell['source']:
            print(line.rstrip())
        break
