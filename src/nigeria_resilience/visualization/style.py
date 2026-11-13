import seaborn as sns


def set_defaults():
    sns.set(rc={'figure.figsize': (16, 8)}, font_scale=1.3)
    sns.set_style("white")


MODEL_PALETTE = {
    "MDN": "#1f77b4",
    "XGB-Mix": "#ff7f0e",
    "NN-1D": "#2ca02c",
    "XGB-1D": "#d62728",
    "Elastic Net": "#9467bd",
    "OLS+GLM": "#8c564b",
}
