"""
plot_fitzpatrick_isb.py
Genera figuras de publicación para el análisis de bias Fitzpatrick — Cohorte ISB.

Figuras generadas:
    1. Barplot MAE por método y grupo Fitzpatrick
    2. Heatmap MAE normalizado (métodos × grupos)
    3. Scatter plots HR predicha vs HR ground truth por grupo (un panel por método)
    4. Barplot Pearson r por método y grupo

Uso:
    python plot_fitzpatrick_isb.py \
        --results_dir results/ISB_Fitzpatrick/ \
        --predictions_dir runs/ISB_Unsupervised/ \
        --gt_csv /ruta/a/ISB_DATASET/isb_ground_truth.csv \
        --output_dir figures/ISB/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

# ── Estilo de publicación ──────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
})

METHODS  = ["ICA", "POS", "CHROM", "GREEN", "LGI", "PBV", "OMIT"]
GROUPS   = ["FP12", "FP3", "FP45"]
G_LABELS = {"FP12": "FP 1+2\n(Claro)", "FP3": "FP 3\n(Intermedio)", "FP45": "FP 4+5\n(Oscuro)"}
G_COLORS = {"FP12": "#378ADD", "FP3": "#639922", "FP45": "#D85A30"}


# ==============================================================================
# Figura 1 — Barplot MAE por método y grupo Fitzpatrick
# ==============================================================================

def plot_mae_barplot(fp_df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_methods = len(METHODS)
    n_groups  = len(GROUPS)
    width     = 0.22
    x         = np.arange(n_methods)

    for i, grp in enumerate(GROUPS):
        grp_data = fp_df[fp_df["fp_group"] == grp].set_index("method")
        maes = [grp_data.loc[m, "MAE"] if m in grp_data.index else np.nan
                for m in METHODS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, maes, width=width,
                      color=G_COLORS[grp], alpha=0.85,
                      label=G_LABELS[grp].replace("\n", " "),
                      edgecolor='white', linewidth=0.5)
        # Etiquetas de valor encima de cada barra
        for bar, val in zip(bars, maes):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.2,
                        f"{val:.1f}", ha='center', va='bottom',
                        fontsize=8, color='#333333')

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("MAE (bpm)")
    ax.set_title("Heart rate estimation error by Fitzpatrick group — ISB cohort (n=50)")
    ax.legend(title="Skin tone group", frameon=False)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.15)
    ax.axhline(y=5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5,
               label='MAE = 5 bpm (reference)')

    plt.tight_layout()
    path = os.path.join(output_dir, "Fig1_MAE_barplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Guardada: {path}")


# ==============================================================================
# Figura 2 — Heatmap MAE normalizado
# ==============================================================================

def plot_mae_heatmap(fp_df, output_dir):
    # Construir matriz métodos × grupos
    matrix = np.zeros((len(METHODS), len(GROUPS)))
    for i, m in enumerate(METHODS):
        for j, g in enumerate(GROUPS):
            row = fp_df[(fp_df["method"] == m) & (fp_df["fp_group"] == g)]
            matrix[i, j] = row["MAE"].values[0] if len(row) > 0 else np.nan

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')

    # Etiquetas de ejes
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([G_LABELS[g].replace("\n", " ") for g in GROUPS])
    ax.set_yticks(range(len(METHODS)))
    ax.set_yticklabels(METHODS)

    # Valores dentro de cada celda
    for i in range(len(METHODS)):
        for j in range(len(GROUPS)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > np.nanmedian(matrix) * 1.3 else 'black'
                ax.text(j, i, f"{val:.1f}", ha='center', va='center',
                        fontsize=10, fontweight='500', color=text_color)

    plt.colorbar(im, ax=ax, label='MAE (bpm)', shrink=0.8)
    ax.set_title("MAE heatmap by method and skin tone group")
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig2_MAE_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Guardada: {path}")


# ==============================================================================
# Figura 3 — Scatter plots HR pred vs HR gt por grupo (panel 7 métodos)
# ==============================================================================

def plot_scatter_panels(all_pred_df, output_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for idx, method in enumerate(METHODS):
        ax = axes[idx]
        df = all_pred_df[all_pred_df["method"] == method]

        for grp in GROUPS:
            mask = df["fitzpatrick_group"] == grp
            sub  = df[mask]
            ax.scatter(sub["hr_gt"], sub["hr_pred"],
                       color=G_COLORS[grp], alpha=0.7, s=40,
                       label=G_LABELS[grp].replace("\n", " "),
                       edgecolors='white', linewidths=0.3)

        # Línea identidad
        lims = [
            min(df["hr_gt"].min(), df["hr_pred"].min()) - 5,
            max(df["hr_gt"].max(), df["hr_pred"].max()) + 5
        ]
        ax.plot(lims, lims, 'k--', linewidth=0.8, alpha=0.5)
        ax.set_xlim(lims); ax.set_ylim(lims)

        # Correlación global
        r, p = stats.pearsonr(df["hr_gt"], df["hr_pred"])
        sig = "p<0.05" if p < 0.05 else f"p={p:.2f}"
        ax.set_title(f"{method}  (r={r:.2f}, {sig})", fontsize=11)
        ax.set_xlabel("HR ground truth (bpm)")
        ax.set_ylabel("HR estimated (bpm)")

    # Leyenda en el panel vacío (posición 7)
    axes[7].axis('off')
    patches = [mpatches.Patch(color=G_COLORS[g],
                              label=G_LABELS[g].replace("\n", " "))
               for g in GROUPS]
    axes[7].legend(handles=patches, title="Skin tone group",
                   loc='center', frameon=False, fontsize=11)

    fig.suptitle("rPPG heart rate estimation vs ground truth — ISB cohort",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig3_scatter_panels.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Guardada: {path}")


# ==============================================================================
# Figura 4 — Pearson r por método y grupo
# ==============================================================================

def plot_pearson_barplot(fp_df, output_dir):
    fig, ax = plt.subplots(figsize=(10, 5))

    n_methods = len(METHODS)
    width     = 0.22
    x         = np.arange(n_methods)

    for i, grp in enumerate(GROUPS):
        grp_data = fp_df[fp_df["fp_group"] == grp].set_index("method")
        rs = [grp_data.loc[m, "Pearson_r"] if m in grp_data.index else np.nan
              for m in METHODS]
        ps = [grp_data.loc[m, "Pearson_p"] if m in grp_data.index else np.nan
              for m in METHODS]
        offset = (i - 1) * width
        bars = ax.bar(x + offset, rs, width=width,
                      color=G_COLORS[grp], alpha=0.85,
                      label=G_LABELS[grp].replace("\n", " "),
                      edgecolor='white', linewidth=0.5)
        # Asterisco si p < 0.05
        for bar, r_val, p_val in zip(bars, rs, ps):
            if not np.isnan(p_val) and p_val < 0.05:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.02,
                        '*', ha='center', va='bottom',
                        fontsize=12, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylabel("Pearson r")
    ax.set_title("Pearson correlation by method and Fitzpatrick group — ISB cohort")
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_ylim(-0.3, 1.1)
    ax.legend(title="Skin tone group", frameon=False)
    ax.text(n_methods - 0.5, 0.52, 'r = 0.5', fontsize=9,
            color='gray', va='bottom')
    ax.text(0.01, 0.97, '* p < 0.05', transform=ax.transAxes,
            fontsize=9, color='gray', va='top')

    plt.tight_layout()
    path = os.path.join(output_dir, "Fig4_Pearson_barplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[OK] Guardada: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Figuras Fitzpatrick — Cohorte ISB")
    parser.add_argument("--results_dir",     required=True,
                        help="Carpeta con ISB_fitzpatrick_metrics.csv y ISB_all_predictions.csv")
    parser.add_argument("--output_dir",      default="figures/ISB/",
                        help="Carpeta de salida para las figuras")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ── Cargar CSVs generados por analyze_fitzpatrick_isb.py ──────────────────
    fp_csv   = os.path.join(args.results_dir, "ISB_fitzpatrick_metrics.csv")
    pred_csv = os.path.join(args.results_dir, "ISB_all_predictions.csv")

    if not os.path.isfile(fp_csv):
        raise FileNotFoundError(f"No se encontró: {fp_csv}\n"
                                "Ejecuta primero analyze_fitzpatrick_isb.py")
    if not os.path.isfile(pred_csv):
        raise FileNotFoundError(f"No se encontró: {pred_csv}\n"
                                "Ejecuta primero analyze_fitzpatrick_isb.py")

    fp_df      = pd.read_csv(fp_csv)
    all_pred   = pd.read_csv(pred_csv)

    print(f"[INFO] Métricas por grupo cargadas: {len(fp_df)} filas")
    print(f"[INFO] Predicciones totales: {len(all_pred)} filas")

    # ── Generar figuras ────────────────────────────────────────────────────────
    print("\nGenerando figuras...")
    plot_mae_barplot(fp_df, args.output_dir)
    plot_mae_heatmap(fp_df, args.output_dir)
    plot_scatter_panels(all_pred, args.output_dir)
    plot_pearson_barplot(fp_df, args.output_dir)

    print(f"\n[INFO] Todas las figuras guardadas en: {args.output_dir}")
    print("       Fig1_MAE_barplot.png")
    print("       Fig2_MAE_heatmap.png")
    print("       Fig3_scatter_panels.png")
    print("       Fig4_Pearson_barplot.png")


if __name__ == "__main__":
    main()
