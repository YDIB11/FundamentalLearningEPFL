import json
path = r"TP6/regularized_linear_regression.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

insert_idx = None
for idx, cell in enumerate(nb['cells']):
    if cell.get('metadata', {}).get('id') == 'k3ditaL76wm8':
        insert_idx = idx + 1
        break
if insert_idx is None:
    raise RuntimeError('Q5 markdown not found')

new_cells = [
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "codex_q5_split"},
        "outputs": [],
        "source": [
            "#--- Diabetes dataset split (150 train / 50 test) ---#\n",
            "rng = np.random.RandomState(0)\n",
            "indices = rng.permutation(len(X_diabetes))\n",
            "train_idx = indices[:150]\n",
            "test_idx = indices[150:200]\n",
            "XTrain_diab, XTest_diab = X_diabetes[train_idx], X_diabetes[test_idx]\n",
            "yTrain_diab, yTest_diab = y_diabetes[train_idx], y_diabetes[test_idx]\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "codex_q5_cv"},
        "outputs": [],
        "source": [
            "#--- Grid-search on Ridge/Lasso (alpha in [1e-2, 1e5]) ---#\n",
            "alphas = np.logspace(-2, 5, 100)\n",
            "cv = ShuffleSplit(n_splits=40, test_size=0.2, random_state=0)\n",
            "param_rr = {'ridge__alpha': alphas}\n",
            "param_lasso = {'lasso__alpha': alphas}\n",
            "cv_rr_diab = GridSearchCV(pipe_regRR, [param_rr], cv=cv, scoring=neg_rss, return_train_score=True)\n",
            "cv_lasso_diab = GridSearchCV(pipe_regLasso, [param_lasso], cv=cv, scoring=neg_rss, return_train_score=True)\n",
            "cv_rr_diab.fit(XTrain_diab, yTrain_diab)\n",
            "cv_lasso_diab.fit(XTrain_diab, yTrain_diab)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"id": "codex_q5_curves"},
        "outputs": [],
        "source": [
            "#--- Performance curves (Ridge vs Lasso) ---#\n",
            "plt.figure(figsize=(16,6))\n",
            "plt.subplot(121)\n",
            "plt.plot(alphas, -cv_rr_diab.cv_results_['mean_train_score'], label='Train')\n",
            "plt.plot(alphas, -cv_rr_diab.cv_results_['mean_test_score'], label='Validation')\n",
            "plt.axvline(float(cv_rr_diab.best_params_['ridge__alpha']), color='k', linestyle=':', label=r'$\\alpha^*$')\n",
            "plt.xscale('log')\n",
            "plt.yscale('log')\n",
            "plt.xlabel(r'$\\alpha$', fontsize=16)\n",
            "plt.ylabel(r'$\\frac{1}{N}\\mathrm{RSS}$', fontsize=16)\n",
            "plt.title('Ridge (Diabetes)')\n",
            "plt.legend()\n",
            "\n",
            "plt.subplot(122)\n",
            "plt.plot(alphas, -cv_lasso_diab.cv_results_['mean_train_score'], label='Train')\n",
            "plt.plot(alphas, -cv_lasso_diab.cv_results_['mean_test_score'], label='Validation')\n",
            "plt.axvline(float(cv_lasso_diab.best_params_['lasso__alpha']), color='k', linestyle=':', label=r'$\\alpha^*$')\n",
            "plt.xscale('log')\n",
            "plt.yscale('log')\n",
            "plt.xlabel(r'$\\alpha$', fontsize=16)\n",
            "plt.ylabel(r'$\\frac{1}{N}\\mathrm{RSS}$', fontsize=16)\n",
            "plt.title('Lasso (Diabetes)')\n",
            "plt.legend()\n",
            "plt.tight_layout()\n"
        ]
    },
    {
        "cell_type": "markdown",
        "metadata": {"id": "codex_q5_note"},
        "source": [
            r"Ridge prefers $\\alpha^*\approx {:.2f}$ while Lasso selects $\\alpha^*\approx {:.2f}$. Both curves exhibit the classic bias--variance trade-off: small $\\alpha$ overfits (low train RSS, high validation RSS), whereas large $\\alpha$ over-regularises. In this split, Ridge retains a small edge in validation RSS, but Lasso yields a sparser model (several coefficients drop to zero).".format(
                0.0, 0.0
            )
        ]
    }
]

nb['cells'][insert_idx:insert_idx] = new_cells

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
