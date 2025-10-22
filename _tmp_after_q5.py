import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
collect = False
for cell in nb['cells']:
    meta = cell.get('metadata', {})
    if meta.get('id') == 'k3ditaL76wm8':
        collect = True
        continue
    if meta.get('id') == 'YX-15tDj3ZJr':
        break
    if collect:
        print('---', cell.get('cell_type'), meta.get('id'))
        for line in cell.get('source', []):
            print(line.rstrip())
        print()
