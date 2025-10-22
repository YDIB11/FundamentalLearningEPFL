import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == '5vsg98jK5Y-2':
        for line in cell.get('source'):
            print(line.rstrip())
        break
