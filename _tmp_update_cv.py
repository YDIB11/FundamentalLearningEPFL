import json
path = r"TP6/regularized_linear_regression.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('metadata', {}).get('id') == 'codex_q5_cv':
        cell['source'] = [
            '#--- Grid-search on Ridge/Lasso (alpha in [1e-2, 1e5]) ---#\n',
            'alphas = np.logspace(-2, 5, 100)\n',
            'cv = ShuffleSplit(n_splits=40, test_size=0.2, random_state=0)\n',
            "param_rr = {'ridge__alpha': alphas}\n",
            "param_lasso = {'lasso__alpha': alphas}\n",
            'cv_rr_diab = GridSearchCV(pipe_regRR, [param_rr], cv=cv, scoring=neg_rss, return_train_score=True)\n',
            'cv_lasso_diab = GridSearchCV(pipe_regLasso, [param_lasso], cv=cv, scoring=neg_rss, return_train_score=True)\n',
            'cv_rr_diab.fit(XTrain_diab, yTrain_diab)\n',
            'cv_lasso_diab.fit(XTrain_diab, yTrain_diab)\n',
            'print(f"Best Ridge alpha: {cv_rr_diab.best_params_[\'ridge__alpha\']:.4g}")\n',
            'print(f"Best Lasso alpha: {cv_lasso_diab.best_params_[\'lasso__alpha\']:.4g}")\n'
        ]
        break
else:
    raise RuntimeError('codex_q5_cv not found')
with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
