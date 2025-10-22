import json
with open("TP6/regularized_linear_regression.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)
for i in range(66,70):
    cell=nb['cells'][i]
    print('Index',i,'type',cell['cell_type'],'id',cell.get('metadata',{}).get('id'))
    for line in cell.get('source', []):
        print(' ',line.rstrip())
    print()
