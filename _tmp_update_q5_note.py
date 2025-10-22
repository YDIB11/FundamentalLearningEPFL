import json
path = r"TP6/regularized_linear_regression.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'codex_q5_note':
        cell['source'] = [
            "Ridge prefers $\\alpha^*\\approx 2.48\\times10^{1}$ while Lasso selects $\\alpha^*\\approx 5.0\\times10^{-1}$. Both curves exhibit the classic bias--variance trade-off: small $\\alpha$ overfits (low train RSS, high validation RSS), whereas large $\\alpha$ over-regularises. In this split, Ridge retains a slight edge in validation RSS, whereas Lasso produces a sparser solution with several coefficients exactly zero." 
        ]
        break
else:
    raise RuntimeError('codex_q5_note not found')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
