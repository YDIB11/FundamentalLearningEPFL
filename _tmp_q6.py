import json
path='TP6/regularized_linear_regression.ipynb'
with open(path,'r',encoding='utf-8') as f:
    nb=json.load(f)

q5_idx=None
for i,cell in enumerate(nb['cells']):
    meta=cell.get('metadata',{})
    if meta.get('id')=='k3ditaL76wm8':
        q5_idx=i
    if meta.get('id')=='YX-15tDj3ZJr':
        insert=i
        break
else:
    insert=len(nb['cells'])

def make_text(lines):
    return [line+'\n' for line in lines]

cells_to_add=[
    {
        'cell_type':'markdown',
        'metadata':{'id':'codex_q6_summary'},
        'source':make_text([
            '**Q6. Behaviour at the extremes of $\\alpha$**',
            '- $\\alpha=0$: the penalty disappears, so Ridge and Lasso both coincide with the ordinary least-squares estimator (with our standardised data the coefficients exactly match OLS).',
            '- $\\alpha\\to\\infty$: the penalty dominates; Ridge shrinks all weights to zero, Lasso forces every coefficient exactly to zero, leaving only the intercept. Consequently both predict the constant average response.'
        ])
    },
    {
        'cell_type':'markdown',
        'metadata':{'id':'codex_q6_path'},
        'source':make_text([
            '**Q6. Qualitative comparison of regularisation paths**',
            '- Ridge paths are smooth: each coefficient decays continuously as $\\alpha$ increases and never reaches exactly zero.',
            '- Lasso paths are piecewise linear with sharp kinks: coefficients drop abruptly to zero and stay there, yielding sparse models. In linear scale this appears as flat segments once a variable exits the model.'
        ])
    }
]

nb['cells'][insert:insert]=cells_to_add

with open(path,'w',encoding='utf-8') as f:
    json.dump(nb,f,indent=1)
