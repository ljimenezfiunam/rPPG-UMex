"""
analyze_fitzpatrick_isb.py
Análisis de bias algorítmico por grupo Fitzpatrick para la cohorte ISB.

Uso:
    python analyze_fitzpatrick_isb.py \
        --results_dir runs/ISB_Unsupervised/ \
        --gt_csv /run/media/lJIMENEZ/datos/Proyectos/NANCI/ISB_DATASET/ISB_DATASET/isb_ground_truth.csv \
        --output_dir results/ISB_Fitzpatrick/
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# ── Métodos disponibles ────────────────────────────────────────────────────────
METHODS = ["ICA", "POS", "CHROM", "GREEN", "LGI", "PBV", "OMIT"]

# ── Grupos Fitzpatrick ─────────────────────────────────────────────────────────
FP_GROUPS = {
    "FP12": [1, 2],
    "FP3":  [3],
    "FP45": [4, 5],
}
FP_LABELS = {"FP12": "FP 1+2 (Claro)", "FP3": "FP 3 (Intermedio)", "FP45": "FP 4+5 (Oscuro)"}


# ==============================================================================
# Métricas
# ==============================================================================

def mae(pred, gt):
    return np.mean(np.abs(pred - gt))

def rmse(pred, gt):
    return np.sqrt(np.mean((pred - gt) ** 2))

def mape(pred, gt):
    mask = gt != 0
    return np.mean(np.abs((pred[mask] - gt[mask]) / gt[mask])) * 100

def pearson(pred, gt):
    if len(pred) < 3:
        return np.nan, np.nan
    r, p = stats.pearsonr(pred, gt)
    return r, p

def spearman(pred, gt):
    if len(pred) < 3:
        return np.nan, np.nan
    r, p = stats.spearmanr(pred, gt)
    return r, p


# ==============================================================================
# Carga de datos
# ==============================================================================

def load_predictions(results_dir, method):
    """Carga el CSV de predicciones de un método."""
    csv_path = os.path.join(results_dir, f"{method}_ISB_predictions.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró: {csv_path}")
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    # Normalizar chunk_id → subject_id (quitar sufijos tipo _chunk0 si existen)
    df["subject_id"] = df["chunk_id"].str.replace(r"_chunk\d+$", "", regex=True)
    return df


def load_ground_truth(gt_csv):
    """Carga el CSV de ground truth con metadatos Fitzpatrick."""
    df = pd.read_csv(gt_csv)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


# ==============================================================================
# Análisis por método
# ==============================================================================

def analyze_method(pred_df, gt_df, method):
    """
    Cruza predicciones con ground truth, calcula métricas globales
    y por grupo Fitzpatrick.

    Returns:
        dict con resultados globales y por grupo
    """
    # Cruzar por subject_id
    merged = pred_df.merge(
        gt_df[["subject_id", "fitzpatrick", "fitzpatrick_group"]],
        on="subject_id",
        how="left"
    )

    missing = merged["fitzpatrick"].isna().sum()
    if missing > 0:
        print(f"  [WARN] {missing} chunks sin match en ground truth")

    merged = merged.dropna(subset=["fitzpatrick"])

    pred  = merged["hr_pred"].values
    gt    = merged["hr_gt"].values
    fp    = merged["fitzpatrick"].values
    fp_grp = merged["fitzpatrick_group"].values

    results = {"method": method, "global": {}, "by_group": {}}

    # ── Métricas globales ──────────────────────────────────────────────────────
    r_p, p_p = pearson(pred, gt)
    r_s, p_s = spearman(pred, gt)
    results["global"] = {
        "n":        len(pred),
        "MAE":      mae(pred, gt),
        "RMSE":     rmse(pred, gt),
        "MAPE":     mape(pred, gt),
        "Pearson_r": r_p,
        "Pearson_p": p_p,
        "Spearman_r": r_s,
        "Spearman_p": p_s,
    }

    # ── Métricas por grupo Fitzpatrick ─────────────────────────────────────────
    for grp_name, fp_types in FP_GROUPS.items():
        mask = np.isin(fp, fp_types)
        p_g  = pred[mask]
        g_g  = gt[mask]

        if len(p_g) < 3:
            results["by_group"][grp_name] = {"n": len(p_g), "MAE": np.nan}
            continue

        r_p_g, p_p_g = pearson(p_g, g_g)
        r_s_g, p_s_g = spearman(p_g, g_g)
        results["by_group"][grp_name] = {
            "n":          len(p_g),
            "MAE":        mae(p_g, g_g),
            "RMSE":       rmse(p_g, g_g),
            "MAPE":       mape(p_g, g_g),
            "Pearson_r":  r_p_g,
            "Pearson_p":  p_p_g,
            "Spearman_r": r_s_g,
            "Spearman_p": p_s_g,
        }

    return results, merged


# ==============================================================================
# Impresión de resultados
# ==============================================================================

def print_results(all_results):
    sep = "=" * 70

    print(f"\n{sep}")
    print("  ANÁLISIS DE BIAS FITZPATRICK — COHORTE ISB")
    print(f"{sep}")

    # ── Tabla global ───────────────────────────────────────────────────────────
    print(f"\n{'Método':<8} {'n':>4} {'MAE':>7} {'RMSE':>7} {'Pearson r':>10} {'p':>8} {'Spearman r':>11} {'p':>8}")
    print("-" * 70)
    for r in all_results:
        g = r["global"]
        print(f"{r['method']:<8} {g['n']:>4} {g['MAE']:>7.2f} {g['RMSE']:>7.2f} "
              f"{g['Pearson_r']:>10.3f} {g['Pearson_p']:>8.4f} "
              f"{g['Spearman_r']:>11.3f} {g['Spearman_p']:>8.4f}")

    # ── Tabla MAE por grupo ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  MAE por grupo Fitzpatrick")
    print(f"{sep}")
    print(f"\n{'Método':<8} {'FP 1+2 (n)':>12} {'FP 3 (n)':>10} {'FP 4+5 (n)':>12}")
    print("-" * 50)
    for r in all_results:
        row = f"{r['method']:<8}"
        for grp in ["FP12", "FP3", "FP45"]:
            g = r["by_group"].get(grp, {})
            mae_val = g.get("MAE", np.nan)
            n_val   = g.get("n", 0)
            row += f"  {mae_val:>6.2f} ({n_val:>2})"
        print(row)

    # ── Pearson por grupo ──────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  Pearson r por grupo Fitzpatrick")
    print(f"{sep}")
    print(f"\n{'Método':<8} {'FP 1+2':>10} {'FP 3':>10} {'FP 4+5':>10}")
    print("-" * 45)
    for r in all_results:
        row = f"{r['method']:<8}"
        for grp in ["FP12", "FP3", "FP45"]:
            g = r["by_group"].get(grp, {})
            r_val = g.get("Pearson_r", np.nan)
            p_val = g.get("Pearson_p", np.nan)
            sig   = "*" if (not np.isnan(p_val) and p_val < 0.05) else " "
            row  += f"  {r_val:>6.3f}{sig:>2}  "
        print(row)
    print("  * p < 0.05")


# ==============================================================================
# Guardar resultados
# ==============================================================================

def save_results(all_results, all_merged, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── CSV global ─────────────────────────────────────────────────────────────
    global_rows = []
    for r in all_results:
        row = {"method": r["method"], **r["global"]}
        global_rows.append(row)
    pd.DataFrame(global_rows).round(4).to_csv(
        os.path.join(output_dir, "ISB_global_metrics.csv"), index=False)

    # ── CSV por grupo ──────────────────────────────────────────────────────────
    group_rows = []
    for r in all_results:
        for grp, metrics in r["by_group"].items():
            row = {"method": r["method"], "fp_group": grp, **metrics}
            group_rows.append(row)
    pd.DataFrame(group_rows).round(4).to_csv(
        os.path.join(output_dir, "ISB_fitzpatrick_metrics.csv"), index=False)

    # ── CSV combinado con predicciones + metadatos ─────────────────────────────
    combined = []
    for r, merged in zip(all_results, all_merged):
        merged["method"] = r["method"]
        combined.append(merged)
    pd.concat(combined, ignore_index=True).to_csv(
        os.path.join(output_dir, "ISB_all_predictions.csv"), index=False)

    print(f"\n[INFO] Resultados guardados en: {output_dir}")
    print(f"       ISB_global_metrics.csv")
    print(f"       ISB_fitzpatrick_metrics.csv")
    print(f"       ISB_all_predictions.csv")


# ==============================================================================
# Main
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Análisis Fitzpatrick — Cohorte ISB")
    parser.add_argument("--results_dir", required=True,
                        help="Carpeta con los CSVs de predicciones (runs/ISB_Unsupervised/)")
    parser.add_argument("--gt_csv", required=True,
                        help="Ruta al archivo isb_ground_truth.csv")
    parser.add_argument("--output_dir", default="results/ISB_Fitzpatrick/",
                        help="Carpeta donde se guardan los resultados")
    args = parser.parse_args()

    # Cargar ground truth
    gt_df = load_ground_truth(args.gt_csv)
    print(f"[INFO] Ground truth cargado: {len(gt_df)} sujetos")

    all_results = []
    all_merged  = []

    for method in METHODS:
        try:
            pred_df = load_predictions(args.results_dir, method)
            print(f"\n[{method}] Chunks cargados: {len(pred_df)}")
            result, merged = analyze_method(pred_df, gt_df, method)
            all_results.append(result)
            all_merged.append(merged)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

    if not all_results:
        print("[ERROR] No se encontraron resultados. Verifica --results_dir")
        return

    print_results(all_results)
    save_results(all_results, all_merged, args.output_dir)


if __name__ == "__main__":
    main()
