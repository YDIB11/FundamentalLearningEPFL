import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb['cells']:
    if 'cv_regRR = GridSearchCV' in ''.join(cell.get('source', [])):
        print(cell.get('source'))
        print('--- id ---', cell.get('metadata', {}).get('id'))
        break
