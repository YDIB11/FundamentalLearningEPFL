import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for i in range(63,71):
    cell=nb['cells'][i]
    print(i, cell.get('cell_type'), cell.get('metadata',{}).get('id'))
