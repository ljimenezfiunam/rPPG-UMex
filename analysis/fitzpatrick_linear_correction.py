"""
fitzpatrick_linear_correction.py
Corrección lineal de la estimación de HR por grupo Fitzpatrick.

Estrategia:
    Para cada método y cada grupo Fitzpatrick, se ajusta una regresión lineal:
        HR_corregida = α × HR_pred + β
    usando leave-one-out cross-validation (LOO-CV) para evitar sobreajuste
    dado el tamaño de muestra limitado (n=16-17 por grupo).

Salidas:
    - Tabla MAE antes vs después de corrección por método y grupo
    - CSV con predicciones corregidas
    - Figura comparativa MAE pre vs post corrección

Uso:
    python fitzpatrick_linear_correction.py \
        --predictions_csv results/ISB_Fitzpatrick/ISB_all_predictions.csv \
        --output_dir results/ISB_Correction/
"""

import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut

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
G_LABELS = {"FP12": "FP 1+2 (Claro)", "FP3": "FP 3 (Intermedio)", "FP45": "FP 4+5 (Oscuro)"}
G_COLORS = {"FP12": "#378ADD", "FP3": "#639922", "FP45": "#D85A30"}


# ==============================================================================
# Corrección LOO-CV por grupo Fitzpatrick
# ==============================================================================

def loo_linear_correction(hr_pred, hr_gt):
    """
    Aplica corrección lineal con Leave-One-Out Cross-Validation.

    Para cada sujeto i:
        1. Entrena regresión lineal con todos los sujetos excepto i
        2. Predice HR corregida para el sujeto i
        3. Calcula el error en el sujeto i

    Esto evita sobreajuste al evaluar la corrección con datos no vistos.

    Args:
        hr_pred (np.ndarray): HR estimada por rPPG
        hr_gt   (np.ndarray): HR ground truth

    Returns:
        np.ndarray: HR corregida por LOO-CV
        float: α promedio (pendiente)
        float: β promedio (intercepto)
    """
    n = len(hr_pred)
    hr_corrected = np.zeros(n)
    alphas = []
    betas  = []

    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(hr_pred):
        X_train = hr_pred[train_idx].reshape(-1, 1)
        y_train = hr_gt[train_idx]
        X_test  = hr_pred[test_idx].reshape(-1, 1)

        model = LinearRegression()
        model.fit(X_train, y_train)

        hr_corrected[test_idx] = model.predict(X_test)
        alphas.append(model.coef_[0])
        betas.append(model.intercept_)

    return hr_corrected, np.mean(alphas), np.mean(betas)


# ==============================================================================
# Análisis principal
# ==============================================================================

def run_correction(df, output_dir):
    """
    Aplica corrección LOO-CV por método y grupo Fitzpatrick.
    Calcula MAE antes y después de corrección.
    """
    results = []
    corrected_rows = []

    for method in METHODS:
        df_m = df[df["method"] == method].copy()

        for grp in GROUPS:
            mask = df_m["fitzpatrick_group"] == grp
            sub  = df_m[mask].copy()

            if len(sub) < 4:
                print(f"[SKIP] {method} / {grp}: n={len(sub)} insuficiente")
                continue

            hr_pred = sub["hr_pred"].values
            hr_gt   = sub["hr_gt"].values

            # MAE antes de corrección
            mae_before = np.mean(np.abs(hr_pred - hr_gt))

            # Corrección LOO-CV
            hr_corr, alpha, beta = loo_linear_correction(hr_pred, hr_gt)

            # MAE después de corrección
            mae_after = np.mean(np.abs(hr_corr - hr_gt))

            improvement = mae_before - mae_after
            pct_improvement = 100 * improvement / mae_before

            results.append({
                "method":      method,
                "fp_group":    grp,
                "n":           len(sub),
                "MAE_before":  round(mae_before, 2),
                "MAE_after":   round(mae_after, 2),
                "improvement": round(improvement, 2),
                "pct_improve": round(pct_improvement, 1),
                "alpha":       round(alpha, 3),
                "beta":        round(beta, 3),
            })

            # Guardar predicciones corregidas
            sub = sub.copy()
            sub["hr_corrected"] = hr_corr
            sub["alpha"] = alpha
            sub["beta"]  = beta
            corrected_rows.append(sub)

    results_df   = pd.DataFrame(results)
    corrected_df = pd.concat(corrected_rows, ignore_index=True)

    return results_df, corrected_df


# ==============================================================================
# Imprimir tabla de resultados
# ==============================================================================

def print_results(results_df):
    print("\n" + "=" * 75)
    print("  CORRECCIÓN LINEAL LOO-CV POR GRUPO FITZPATRICK — COHORTE ISB")
    print("=" * 75)
    print(f"\n{'Método':<8} {'Grupo':<10} {'n':>3} {'MAE antes':>10} "
          f"{'MAE después':>12} {'Mejora':>8} {'%':>7} {'α':>7} {'β':>8}")
    print("-" * 75)

    for method in METHODS:
        sub = results_df[results_df["method"] == method]
        for _, row in sub.iterrows():
            arrow = "↓" if row["improvement"] > 0 else "↑"
            print(f"{row['method']:<8} {row['fp_group']:<10} {row['n']:>3} "
                  f"{row['MAE_before']:>10.2f} {row['MAE_after']:>12.2f} "
                  f"{row['improvement']:>7.2f}{arrow} "
                  f"{row['pct_improve']:>6.1f}% "
                  f"{row['alpha']:>7.3f} {row['beta']:>8.3f}")
        print()

    # Resumen global
    print("=" * 75)
    print("  RESUMEN GLOBAL")
    print("=" * 75)
    summary = results_df.groupby("method").agg(
        MAE_before=("MAE_before", "mean"),
        MAE_after=("MAE_after", "mean"),
        pct_improve=("pct_improve", "mean")
    ).round(2)
    print(f"\n{'Método':<8} {'MAE antes':>10} {'MAE después':>12} {'% mejora':>10}")
    print("-" * 45)
    for method, row in summary.iterrows():
        print(f"{method:<8} {row['MAE_before']:>10.2f} "
              f"{row['MAE_after']:>12.2f} {row['pct_improve']:>9.1f}%")


# ==============================================================================
# Figura comparativa MAE antes vs después
# ==============================================================================

def plot_correction(results_df, output_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

    for ax, grp in zip(axes, GROUPS):
        sub = results_df[results_df["fp_group"] == grp]
        if sub.empty:
            continue

        x     = np.arange(len(METHODS))
        width = 0.35

        mae_before = [sub[sub["method"] == m]["MAE_before"].values[0]
                      if len(sub[sub["method"] == m]) > 0 else np.nan
                      for m in METHODS]
        mae_after  = [sub[sub["method"] == m]["MAE_after"].values[0]
                      if len(sub[sub["method"] == m]) > 0 else np.nan
                      for m in METHODS]

        bars1 = ax.bar(x - width/2, mae_before, width, label="Sin corrección",
                       color=G_COLORS[grp], alpha=0.5, edgecolor='white')
        bars2 = ax.bar(x + width/2, mae_after,  width, label="Con corrección",
                       color=G_COLORS[grp], alpha=0.95, edgecolor='white')

        # Etiquetas
        for bar, val in zip(bars1, mae_before):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha='center', va='bottom', fontsize=8)
        for bar, val in zip(bars2, mae_after):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f"{val:.1f}", ha='center', va='bottom', fontsize=8)

        ax.set_title(G_LABELS[grp])
        ax.set_xticks(x)
        ax.set_xticklabels(METHODS, rotation=45, ha='right')
        ax.set_ylabel("MAE (bpm)")
        ax.axhline(y=5, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
        ax.legend(frameon=False, fontsize=9)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.2)

    fig.suptitle("MAE before and after linear correction by Fitzpatrick group — ISB cohort",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, "Fig5_MAE_correction.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n[OK] Guardada: {path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Corrección lineal LOO-CV por grupo Fitzpatrick — ISB")
    parser.add_argument("--predictions_csv", required=True,
                        help="ISB_all_predictions.csv generado por analyze_fitzpatrick_isb.py")
    parser.add_argument("--output_dir", default="results/ISB_Correction/",
                        help="Carpeta de salida")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Cargar predicciones
    df = pd.read_csv(args.predictions_csv)
    print(f"[INFO] Predicciones cargadas: {len(df)} filas, "
          f"{df['method'].nunique()} métodos, {df['subject_id'].nunique()} sujetos")

    # Aplicar corrección
    results_df, corrected_df = run_correction(df, args.output_dir)

    # Imprimir resultados
    print_results(results_df)

    # Guardar CSVs
    results_df.to_csv(
        os.path.join(args.output_dir, "ISB_correction_metrics.csv"), index=False)
    corrected_df.to_csv(
        os.path.join(args.output_dir, "ISB_corrected_predictions.csv"), index=False)
    print(f"\n[INFO] CSVs guardados en: {args.output_dir}")

    # Generar figura
    plot_correction(results_df, args.output_dir)


if __name__ == "__main__":
    main()
