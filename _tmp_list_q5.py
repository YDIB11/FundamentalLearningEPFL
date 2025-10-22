import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for idx, cell in enumerate(nb['cells']):
    mid = cell.get('metadata', {}).get('id')
    if mid == 'codex_q5_discussion':
        # collect context around
        for j in range(idx-5, idx+5):
            if 0 <= j < len(nb['cells']):
                c = nb['cells'][j]
                print(j, c.get('cell_type'), c.get('metadata', {}).get('id'))
        break
