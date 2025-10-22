import json
path="TP6/regularized_linear_regression.ipynb"
with open(path,'r',encoding='utf-8') as f:
    nb=json.load(f)

# Set metadata IDs
nb['cells'][67]['metadata']={'id':'codex_q6_summary'}
nb['cells'][68]['metadata']={'id':'codex_q6_path_text'}

# remove trailing empty cell if blank
del nb['cells'][69]

with open(path,'w',encoding='utf-8') as f:
    json.dump(nb,f,indent=1)
