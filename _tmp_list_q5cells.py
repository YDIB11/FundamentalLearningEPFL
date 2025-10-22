import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for idx, cell in enumerate(nb['cells']):
    mid = cell.get('metadata', {}).get('id')
    if mid and mid.startswith('codex_q5'):
        print(idx, cell.get('cell_type'), mid)
