import json
path = r"TP6/regularized_linear_regression.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell = nb['cells'][60]
cell['cell_type'] = 'code'
cell['metadata'] = {'id': 'codex_q5_load'}
cell['execution_count'] = None
cell['outputs'] = []
cell['source'] = [
    'from sklearn.datasets import load_diabetes\n',
    '\n',
    '#--- Load Diabetes dataset ---#\n',
    'diabetes = load_diabetes()\n',
    'X_diabetes = diabetes.data\n',
    'y_diabetes = diabetes.target\n'
]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
