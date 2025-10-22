import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for idx in range(55, 70):
    if idx < len(nb['cells']):
        cell = nb['cells'][idx]
        print('Index', idx, cell.get('cell_type'), cell.get('metadata', {}).get('id'))
        for line in cell.get('source', [])[:3]:
            print('  ', line.rstrip())
        if len(cell.get('source', [])) > 3:
            print('   ...')
        print()
