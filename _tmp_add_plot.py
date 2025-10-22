import json
path="TP6/regularized_linear_regression.ipynb"
with open(path,'r',encoding='utf-8') as f:
    nb=json.load(f)

insert_idx=67

code_cell={
    'cell_type':'code',
    'execution_count':None,
    'metadata':{'id':'codex_q6_path_plot'},
    'outputs':[],
    'source':[
        "#--- Ridge vs Lasso coefficient paths (linear alpha scale) ---#\n",
        "scaler_diab = preprocessing.StandardScaler().fit(XTrain_diab)\n",
        "XTrain_diab_scaled = scaler_diab.transform(XTrain_diab)\n",
        "alphas_linear = np.linspace(1e-3, 60, 60)\n",
        "ridge_path = []\n",
        "lasso_path = []\n",
        "for a in alphas_linear:\n",
        "    ridge = linear_model.Ridge(alpha=a)\n",
        "    ridge.fit(XTrain_diab_scaled, yTrain_diab)\n",
        "    ridge_path.append(ridge.coef_)\n",
        "\n",
        "    lasso = linear_model.Lasso(alpha=a, max_iter=20000)\n",
        "    lasso.fit(XTrain_diab_scaled, yTrain_diab)\n",
        "    lasso_path.append(lasso.coef_)\n",
        "\n",
        "ridge_path = np.array(ridge_path)\n",
        "lasso_path = np.array(lasso_path)\n",
        "feature_names = diabetes.feature_names\n",
        "\n",
        "fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)\n",
        "for idx, name in enumerate(feature_names):\n",
        "    axes[0].plot(alphas_linear, ridge_path[:, idx], label=name)\n",
        "axes[0].set_xlabel(r'$\\alpha$')\n",
        "axes[0].set_ylabel('Coefficient value')\n",
        "axes[0].set_title('Ridge path (linear scale)')\n",
        "axes[0].axvline(float(cv_rr_diab.best_params_['ridge__alpha']), color='k', linestyle=':')\n",
        "for idx, name in enumerate(feature_names):\n",
        "    axes[1].plot(alphas_linear, lasso_path[:, idx], label=name)\n",
        "axes[1].set_xlabel(r'$\\alpha$')\n",
        "axes[1].set_title('Lasso path (linear scale)')\n",
        "axes[1].axvline(float(cv_lasso_diab.best_params_['lasso__alpha']), color='k', linestyle=':')\n",
        "axes[1].legend(loc='upper right', fontsize=8, ncol=2)\n",
        "plt.tight_layout()\n"
    ]
}

nb['cells'].insert(insert_idx, code_cell)

with open(path,'w',encoding='utf-8') as f:
    json.dump(nb,f,indent=1)
