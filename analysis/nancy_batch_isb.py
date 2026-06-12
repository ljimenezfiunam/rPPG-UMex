"""
nancy_batch_isb.py  (v2 — sin MediaPipe)
=========================================
Procesamiento batch de los 50 videos ISB para estimar:
  - Presión arterial (SBP, DBP) via PTT frente+cuello
  - Frecuencia respiratoria via modulación de amplitud
  - Frecuencia cardíaca via CHROM (validación)

Detección de ROIs: solo OpenCV (Haar Cascade + landmarks geométricos).
No requiere MediaPipe.

Uso:
    conda activate nancy   # o rppg-toolbox
    python nancy_batch_isb.py \
        --video_dir  /ruta/a/ISB_DATASET/ \
        --gt_csv     /ruta/a/ISB_DATASET/isb_ground_truth.csv \
        --output_dir results/ISB_NANCY/
"""

import os
import argparse
import warnings
warnings.filterwarnings('ignore')

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal, stats
from scipy.signal import find_peaks, butter, sosfiltfilt
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
FP_GROUPS  = ['FP12', 'FP3', 'FP45']
FP_COLORS  = {'FP12': '#378ADD', 'FP3': '#639922', 'FP45': '#D85A30'}
FP_LABELS  = {'FP12': 'FP 1+2 (Claro)', 'FP3': 'FP 3 (Intermedio)', 'FP45': 'FP 4+5 (Oscuro)'}

SKIN_TONE_FACTORS = {1:0.90, 2:0.95, 3:1.00, 4:1.05, 5:1.10, 6:1.15}
BP_ADJUSTMENTS    = {1:(-2,-1), 2:(-1,-0.5), 3:(0,0), 4:(1,0.5), 5:(2,1), 6:(3,1.5)}

plt.rcParams.update({
    'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight',
    'axes.spines.top':False,'axes.spines.right':False,
})


# =============================================================================
# DETECCIÓN DE ROIs (solo OpenCV)
# =============================================================================

def init_detectors():
    """Inicializa detectores Haar para rostro y ojos."""
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade  = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml')
    return face_cascade, eye_cascade


def detectar_rois(frame, face_cascade):
    """
    Detecta ROI de frente y cuello a partir del rostro detectado por Haar.

    ROI frente: tercio superior del bounding box del rostro
    ROI cuello: región inmediatamente debajo del rostro

    Returns:
        roi_frente (tuple): (x, y, w, h) o None
        roi_cuello (tuple): (x, y, w, h) o None
    """
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        return None, None

    # Rostro más grande
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    h_img, w_img = frame.shape[:2]

    # ── ROI Frente: tercio superior del rostro, centrado ──────────────
    f_h   = int(h * 0.25)           # altura de la ROI frontal
    f_w   = int(w * 0.6)            # ancho
    f_x   = x + int((w - f_w) / 2)
    f_y   = y + int(h * 0.05)       # pequeño margen desde el tope
    f_x   = max(0, min(f_x, w_img - f_w))
    f_y   = max(0, min(f_y, h_img - f_h))
    roi_frente = (f_x, f_y, f_w, f_h) if f_w > 20 and f_h > 10 else None

    # ── ROI Cuello: debajo del rostro ─────────────────────────────────
    c_y1  = y + h
    c_y2  = min(y + int(h * 1.55), h_img)
    c_w   = int(w * 0.7)
    c_x   = x + int((w - c_w) / 2)
    c_x   = max(0, min(c_x, w_img - c_w))
    c_h   = c_y2 - c_y1
    roi_cuello = (c_x, c_y1, c_w, c_h) if c_w > 20 and c_h > 20 else None

    return roi_frente, roi_cuello


# =============================================================================
# EXTRACCIÓN DE SEÑALES
# =============================================================================

def extraer_valor_roi(frame, roi, skin_factor, senal_hist):
    """Extrae valor PPG de una ROI rectangular (canal Cr × R normalizado)."""
    x, y, w, h = roi
    h_img, w_img = frame.shape[:2]
    x = max(0, min(x, w_img-1)); y = max(0, min(y, h_img-1))
    w = max(1, min(w, w_img-x)); h = max(1, min(h, h_img-y))

    region = frame[y:y+h, x:x+w]
    if region.size == 0:
        return 0.0

    ycrcb = cv2.cvtColor(region, cv2.COLOR_BGR2YCrCb)
    cr    = float(cv2.mean(ycrcb)[1])
    r     = float(cv2.mean(region)[2])
    val   = cr * (r / (r + 30)) * skin_factor

    if len(senal_hist) > 50:
        arr = np.array(senal_hist[-50:])
        q10, q90 = np.percentile(arr, [10, 90])
        if q90 - q10 > 1e-5:
            val = (val - q10) / (q90 - q10)

    return float(val)


def extraer_rgb_roi(frame, roi):
    """Extrae canales RGB medios de una ROI rectangular."""
    x, y, w, h = roi
    region = frame[y:y+h, x:x+w]
    if region.size == 0:
        return 0.0, 0.0, 0.0
    mean = cv2.mean(region)
    return float(mean[2]), float(mean[1]), float(mean[0])  # R, G, B


# =============================================================================
# PTT Y BP
# =============================================================================

def calcular_ptt(senal_frente, senal_cuello, fps):
    """PTT por correlación cruzada entre señales de frente y cuello."""
    n = min(100, len(senal_frente), len(senal_cuello))
    if n < 30:
        return None

    sf = np.array(senal_frente[-n:], dtype=float)
    sc = np.array(senal_cuello[-n:], dtype=float)
    sf = (sf - np.mean(sf)) / (np.std(sf) + 1e-5)
    sc = (sc - np.mean(sc)) / (np.std(sc) + 1e-5)

    corr = np.correlate(sf, sc, mode='full')
    lags = np.arange(-n + 1, n)
    ptt  = abs(lags[np.argmax(corr)]) / fps

    return ptt if 0.04 <= ptt <= 0.25 else None


def estimar_bp(ptt, skin_factor, adj_sbp, adj_dbp):
    """Estima SBP y DBP desde PTT con ajuste por tono de piel."""
    ptt_adj = ptt * skin_factor
    sbp = float(np.clip(120 - 80 * ptt_adj + adj_sbp, 80, 200))
    dbp = float(np.clip(80  - 50 * ptt_adj + adj_dbp, 40, 120))
    if sbp <= dbp:
        dbp = sbp - 10
    return sbp, dbp


# =============================================================================
# FRECUENCIA RESPIRATORIA
# =============================================================================

def estimar_fr(senal_frente, fps):
    """
    Estima FR desde la modulación de amplitud de la señal de frente.

    Método:
        1. Filtrar en banda cardíaca para obtener pulso limpio
        2. Envolvente via transformada de Hilbert
        3. FFT de la envolvente en banda respiratoria (0.13–0.50 Hz)

    A diferencia del rPPG-Toolbox (que aplica bandpass 0.7-3 Hz y elimina
    la banda respiratoria), este método extrae la modulación de amplitud
    del pulso cardíaco que sí lleva información respiratoria.
    """
    n = len(senal_frente)
    if n < int(fps * 15):
        return np.nan

    sig = np.array(senal_frente, dtype=float)
    sig -= np.mean(sig)

    # Paso 1: filtrar en banda cardíaca
    try:
        sos_hr = butter(4, [0.7, 3.0], btype='band', output='sos', fs=fps)
        sig_hr = sosfiltfilt(sos_hr, sig)
    except Exception:
        sig_hr = sig

    # Paso 2: envolvente (Hilbert)
    envolvente = np.abs(signal.hilbert(sig_hr))

    # Paso 3: suavizar envolvente
    try:
        sos_env = butter(2, 0.5, btype='low', output='sos', fs=fps)
        envolvente = sosfiltfilt(sos_env, envolvente)
    except Exception:
        pass

    # Paso 4: FFT en banda respiratoria
    envolvente -= np.mean(envolvente)
    freqs    = np.fft.rfftfreq(n, d=1.0 / fps)
    spectrum = np.abs(np.fft.rfft(envolvente * np.hanning(n)))
    valid    = (freqs >= 0.13) & (freqs <= 0.50)

    if not np.any(valid):
        return np.nan

    return float(freqs[valid][np.argmax(spectrum[valid])] * 60.0)


# =============================================================================
# PROCESAMIENTO DE UN VIDEO
# =============================================================================

def procesar_video(video_path, fitzpatrick, face_cascade, signal_dir=None):
    """Procesa un video completo y retorna estimaciones de BP, FR y HR."""
    skin_factor     = SKIN_TONE_FACTORS.get(fitzpatrick, 1.0)
    adj_sbp, adj_dbp = BP_ADJUSTMENTS.get(fitzpatrick, (0, 0))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    senal_frente, senal_cuello = [], []
    rgb_r, rgb_g, rgb_b = [], [], []
    sbp_vals, dbp_vals, ptt_hist = [], [], []

    roi_frente_cache = None
    roi_cuello_cache = None
    detect_interval  = 30   # re-detectar cada 30 frames
    frame_idx        = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        # ── Detectar ROIs cada N frames ────────────────────────────────
        if frame_idx % detect_interval == 1 or roi_frente_cache is None:
            rf, rc = detectar_rois(frame, face_cascade)
            if rf is not None:
                roi_frente_cache = rf
            if rc is not None:
                roi_cuello_cache = rc

        # ── Extraer señales ────────────────────────────────────────────
        if roi_frente_cache is not None:
            val_f = extraer_valor_roi(frame, roi_frente_cache,
                                      skin_factor, senal_frente)
            senal_frente.append(val_f)
            r, g, b = extraer_rgb_roi(frame, roi_frente_cache)
            rgb_r.append(r); rgb_g.append(g); rgb_b.append(b)

        if roi_cuello_cache is not None:
            val_c = extraer_valor_roi(frame, roi_cuello_cache,
                                      skin_factor, senal_cuello)
            senal_cuello.append(val_c)

        # ── Estimar PTT y BP cada 30 frames ────────────────────────────
        if (frame_idx % 30 == 0 and
                len(senal_frente) > 30 and
                len(senal_cuello) > 30):
            ptt = calcular_ptt(senal_frente, senal_cuello, fps)
            if ptt is not None:
                sbp, dbp = estimar_bp(ptt, skin_factor, adj_sbp, adj_dbp)
                sbp_vals.append(sbp)
                dbp_vals.append(dbp)
                ptt_hist.append(ptt)

    cap.release()

    # ── Guardar señales crudas ─────────────────────────────────────────
    if signal_dir:
        subj = os.path.splitext(os.path.basename(video_path))[0]
        np.savez_compressed(
            os.path.join(signal_dir, f"{subj}_signals.npz"),
            senal_frente=np.array(senal_frente),
            senal_cuello=np.array(senal_cuello),
            ptt=np.array(ptt_hist),
            sbp=np.array(sbp_vals),
            dbp=np.array(dbp_vals),
        )

    # ── HR via CHROM ───────────────────────────────────────────────────
    hr_est = np.nan
    if len(rgb_g) > int(fps * 10):
        try:
            r_a = np.array(rgb_r, dtype=float)
            g_a = np.array(rgb_g, dtype=float)
            b_a = np.array(rgb_b, dtype=float)
            xs    = 3*r_a - 2*g_a
            ys    = 1.5*r_a + g_a - 1.5*b_a
            alpha = np.std(xs) / (np.std(ys) + 1e-5)
            chrom = xs - alpha * ys
            chrom -= np.mean(chrom)
            freqs = np.fft.rfftfreq(len(chrom), d=1.0/fps)
            spec  = np.abs(np.fft.rfft(chrom * np.hanning(len(chrom))))
            valid = (freqs >= 0.7) & (freqs <= 3.0)
            if np.any(valid):
                hr_est = float(freqs[valid][np.argmax(spec[valid])] * 60.0)
        except Exception:
            pass

    return {
        'n_frames':         frame_idx,
        'fps':              round(fps, 1),
        'cuello_detectado': roi_cuello_cache is not None,
        'ptt_validos':      len(ptt_hist),
        'ptt_mean':         round(float(np.mean(ptt_hist)), 4) if ptt_hist else np.nan,
        'sbp_nancy':        round(float(np.mean(sbp_vals)), 1) if sbp_vals else np.nan,
        'sbp_std':          round(float(np.std(sbp_vals)),  1) if sbp_vals else np.nan,
        'dbp_nancy':        round(float(np.mean(dbp_vals)), 1) if dbp_vals else np.nan,
        'dbp_std':          round(float(np.std(dbp_vals)),  1) if dbp_vals else np.nan,
        'fr_nancy':         round(estimar_fr(senal_frente, fps), 2)
                            if len(senal_frente) > int(fps*15) else np.nan,
        'hr_nancy':         round(hr_est, 1) if not np.isnan(hr_est) else np.nan,
    }


# =============================================================================
# ANÁLISIS Y FIGURAS
# =============================================================================

def print_correlations(df):
    print("\n" + "="*65)
    print("  CORRELACIONES NANCY vs GROUND TRUTH — ISB cohort")
    print("="*65)

    pairs = [('sbp_nancy','bp_sys','SBP (mmHg)'),
             ('dbp_nancy','bp_dia','DBP (mmHg)'),
             ('fr_nancy', 'fr_gt', 'FR  (rpm) '),
             ('hr_nancy', 'hr_gt', 'HR  (bpm) ')]

    print(f"\n{'Variable':<12} {'n':>4} {'MAE':>7} {'Pearson r':>10} {'p':>8} {'Sig':>5}")
    print("-" * 52)
    for est, gt, label in pairs:
        sub = df.dropna(subset=[est, gt])
        n   = len(sub)
        if n < 5:
            print(f"{label:<12} {n:>4}  (insuficiente)")
            continue
        mae  = np.mean(np.abs(sub[est] - sub[gt]))
        r, p = stats.pearsonr(sub[est], sub[gt])
        sig  = '**' if p<0.01 else ('*' if p<0.05 else 'n.s.')
        print(f"{label:<12} {n:>4} {mae:>7.2f} {r:>10.3f} {p:>8.4f} {sig:>5}")

    print(f"\n{'Variable':<8} {'Grupo':<12} {'n':>4} {'MAE':>7} {'Pearson r':>10} {'p':>8}")
    print("-" * 58)
    for est, gt, label in [('sbp_nancy','bp_sys','SBP'),
                            ('dbp_nancy','bp_dia','DBP'),
                            ('fr_nancy', 'fr_gt', 'FR' )]:
        for grp in FP_GROUPS:
            sub = df[df['fp_group']==grp].dropna(subset=[est,gt])
            n   = len(sub)
            if n < 4:
                continue
            mae  = np.mean(np.abs(sub[est] - sub[gt]))
            r, p = stats.pearsonr(sub[est], sub[gt])
            sig  = '*' if p<0.05 else 'n.s.'
            print(f"{label:<8} {grp:<12} {n:>4} {mae:>7.2f} {r:>10.3f} {p:>8.4f}  {sig}")
        print()


def plot_scatter(df, est_col, gt_col, xlabel, ylabel, title, out_path):
    valid = df.dropna(subset=[est_col, gt_col])
    fig, ax = plt.subplots(figsize=(7, 6))
    for grp in FP_GROUPS:
        s = valid[valid['fp_group']==grp]
        ax.scatter(s[gt_col], s[est_col], label=FP_LABELS[grp],
                   color=FP_COLORS[grp], alpha=0.75, s=60,
                   edgecolors='white', linewidths=0.3)
    if len(valid) > 5:
        m, b, r, p, _ = stats.linregress(valid[gt_col], valid[est_col])
        xl = np.linspace(valid[gt_col].min(), valid[gt_col].max(), 100)
        ax.plot(xl, m*xl+b, 'k--', lw=1.5,
                label=f'Regression (r={r:.2f}, p={p:.3f})')
    lims = [min(valid[gt_col].min(), valid[est_col].min())-5,
            max(valid[gt_col].max(), valid[est_col].max())+5]
    ax.plot(lims, lims, color='gray', lw=0.8, alpha=0.4, label='Identity')
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend(frameon=False, fontsize=9)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[OK] {out_path}')


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir',    required=True)
    parser.add_argument('--gt_csv',       required=True)
    parser.add_argument('--output_dir',   default='results/ISB_NANCY/')
    parser.add_argument('--save_signals', action='store_true')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    signal_dir = None
    if args.save_signals:
        signal_dir = os.path.join(args.output_dir, 'signals')
        os.makedirs(signal_dir, exist_ok=True)

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt = pd.read_csv(args.gt_csv)
    gt.columns = [c.strip().lower() for c in gt.columns]
    if 'fitzpatrick_group' in gt.columns:
        gt.rename(columns={'fitzpatrick_group':'fp_group'}, inplace=True)
    gt['bp_sys'] = gt[['bp_sys_1','bp_sys_2']].mean(axis=1)
    gt['bp_dia'] = gt[['bp_dia_1','bp_dia_2']].mean(axis=1)
    gt.set_index('subject_id', inplace=True)
    print(f"[INFO] Ground truth: {len(gt)} sujetos")

    # ── Detectores ────────────────────────────────────────────────────────────
    face_cascade, _ = init_detectors()

    # ── Sujetos ───────────────────────────────────────────────────────────────
    subjects = sorted(
        [d for d in os.listdir(args.video_dir)
         if os.path.isdir(os.path.join(args.video_dir, d))
         and d.startswith('Sujeto')],
        key=lambda x: int(x.replace('Sujeto',''))
    )
    print(f"[INFO] Sujetos encontrados: {len(subjects)}")

    results = []
    for subj in tqdm(subjects, desc="NANCY batch"):
        video_path = os.path.join(args.video_dir, subj, f"{subj}.mp4")
        if not os.path.isfile(video_path):
            print(f"\n[SKIP] {video_path} no existe")
            continue
        if subj not in gt.index:
            print(f"\n[SKIP] Sin GT para {subj}")
            continue

        row  = gt.loc[subj]
        fitz = int(row['fitzpatrick'])

        try:
            res = procesar_video(video_path, fitz, face_cascade, signal_dir)
            if res is None:
                continue
            res.update({
                'subject_id': subj,
                'fitzpatrick': fitz,
                'fp_group':   str(row['fp_group']),
                'hr_gt':      float(row['hr_gt']),
                'fr_gt':      float(row['fr_gt']),
                'bp_sys':     float(row['bp_sys']),
                'bp_dia':     float(row['bp_dia']),
            })
            results.append(res)
        except Exception as e:
            print(f"\n[ERROR] {subj}: {e}")

    if not results:
        print("[ERROR] No se procesó ningún sujeto.")
        return

    # ── Guardar CSV ───────────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    col_order = ['subject_id','fitzpatrick','fp_group','fps','n_frames',
                 'cuello_detectado','ptt_validos','ptt_mean',
                 'sbp_nancy','sbp_std','dbp_nancy','dbp_std',
                 'fr_nancy','hr_nancy',
                 'hr_gt','fr_gt','bp_sys','bp_dia']
    df = df[[c for c in col_order if c in df.columns]]
    out_csv = os.path.join(args.output_dir, 'ISB_nancy_results.csv')
    df.to_csv(out_csv, index=False)

    print(f"\n[INFO] Sujetos procesados : {len(df)}")
    print(f"[INFO] Con cuello detectado: {df['cuello_detectado'].sum()}")
    print(f"[INFO] PTT válidos promedio: {df['ptt_validos'].mean():.1f}")

    # ── Correlaciones ─────────────────────────────────────────────────────────
    print_correlations(df)

    # ── Figuras ───────────────────────────────────────────────────────────────
    print("\n[INFO] Generando figuras...")
    plot_scatter(df, 'sbp_nancy', 'bp_sys',
                 'SBP ground truth (mmHg)', 'SBP NANCY estimate (mmHg)',
                 'Systolic BP estimation via PTT — ISB cohort',
                 os.path.join(fig_dir, 'Fig_SBP_scatter.png'))
    plot_scatter(df, 'dbp_nancy', 'bp_dia',
                 'DBP ground truth (mmHg)', 'DBP NANCY estimate (mmHg)',
                 'Diastolic BP estimation via PTT — ISB cohort',
                 os.path.join(fig_dir, 'Fig_DBP_scatter.png'))
    plot_scatter(df, 'fr_nancy', 'fr_gt',
                 'FR ground truth (rpm)', 'FR NANCY estimate (rpm)',
                 'Respiratory rate via amplitude modulation — ISB cohort',
                 os.path.join(fig_dir, 'Fig_FR_scatter.png'))
    plot_scatter(df, 'hr_nancy', 'hr_gt',
                 'HR ground truth (bpm)', 'HR NANCY estimate (bpm)',
                 'Heart rate via CHROM (validation) — ISB cohort',
                 os.path.join(fig_dir, 'Fig_HR_scatter.png'))

    print("\n✅ NANCY batch completado.")
    print(f"   Resultados: {out_csv}")


if __name__ == '__main__':
    main()
