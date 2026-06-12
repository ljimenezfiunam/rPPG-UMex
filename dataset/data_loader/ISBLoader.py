"""
ISBLoader.py
DataLoader para la cohorte ISB (Instituto de Señales Biomédicas, CDMX) compatible
con rPPG-Toolbox.

Estructura esperada del directorio:
    /path/to/ISB_DATASET/
    ├── Sujeto1/
    │   └── Sujeto1.mp4
    ├── Sujeto2/
    │   └── Sujeto2.mp4
    ├── ...
    └── Sujeto50/
        └── Sujeto50.mp4

Ground truth (HR, FR, BP, Fitzpatrick) se carga desde un CSV externo con la
siguiente estructura de columnas:
    subject_id, fitzpatrick, fitzpatrick_group, gender, age,
    hr_gt, fr_gt, bp_sys_1, bp_dia_1, bp_sys_2, bp_dia_2

Uso:
    1. Copiar este archivo a:
         rPPG-Toolbox/dataset/data_loader/ISBLoader.py
    2. Registrar la clase en:
         rPPG-Toolbox/dataset/data_loader/__init__.py
    3. Añadir "ISB" al dispatcher de datasets en main.py (igual que VITALVIDEOS)
    4. Crear el YAML de configuración (ver ISB_UNSUPERVISED.yaml)
"""

import os
import sys

_TOOLBOX_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _TOOLBOX_ROOT not in sys.path:
    sys.path.insert(0, _TOOLBOX_ROOT)

import cv2
import numpy as np
import pandas as pd
from dataset.data_loader.BaseLoader import BaseLoader
from tqdm import tqdm


class ISBLoader(BaseLoader):
    """
    DataLoader para la cohorte ISB de estudiantes universitarios de CDMX.

    Características de la cohorte:
        - 50 sujetos (Sujeto1 … Sujeto50)
        - Distribución Fitzpatrick: FP1+2 = 16, FP3 = 17, FP4+5 = 17
        - Ground truth: HR (sensor de presión), FR (respiratoria), BP (sys/dia)
        - Un video por sujeto, adquirido en condiciones controladas

    Grupos de análisis de bias:
        FP12  → Fitzpatrick tipos 1 y 2 (tonos claros)
        FP3   → Fitzpatrick tipo 3 (tono intermedio)
        FP45  → Fitzpatrick tipos 4 y 5 (tonos oscuros)
    """

    def __init__(self, name, data_path, config_data):
        super().__init__(name, data_path, config_data)

    # ------------------------------------------------------------------
    # Métodos requeridos por BaseLoader
    # ------------------------------------------------------------------

    def get_raw_data(self, data_path):
        """
        Escanea ISB_DATASET/ y construye la lista de ítems procesables.

        Cada ítem incluye la ruta al video y todos los ground truth disponibles.
        El CSV de ground truth debe estar en data_path/isb_ground_truth.csv
        (o en la ruta definida en GT_PATH del YAML).

        Returns:
            list[dict]: cada elemento contiene:
                - "index"          : "SujetoN"
                - "path"           : ruta a la carpeta del sujeto
                - "video_filename" : "SujetoN.mp4"
                - "hr_gt"          : frecuencia cardíaca ground truth (bpm)
                - "fr_gt"          : frecuencia respiratoria ground truth (rpm)
                - "bp_sys"         : presión sistólica media (mmHg) o None
                - "bp_dia"         : presión diastólica media (mmHg) o None
                - "fitzpatrick"    : tipo Fitzpatrick (1-5)
                - "fp_group"       : "FP12", "FP3" o "FP45"
                - "gender"         : "Masculino" o "Femenino"
                - "age"            : edad en años
        """
        # ── Cargar CSV de ground truth ──────────────────────────────────
        gt_csv = os.path.join(data_path, "isb_ground_truth.csv")
        if not os.path.isfile(gt_csv):
            raise FileNotFoundError(
                f"No se encontró el CSV de ground truth en:\n  {gt_csv}\n"
                "Asegúrate de colocar 'isb_ground_truth.csv' dentro de DATA_PATH."
            )

        gt_df = pd.read_csv(gt_csv)
        gt_df.set_index("subject_id", inplace=True)

        # ── Calcular BP promedio (primera y última medición) ─────────────
        gt_df["bp_sys_mean"] = gt_df[["bp_sys_1", "bp_sys_2"]].mean(axis=1)
        gt_df["bp_dia_mean"] = gt_df[["bp_dia_1", "bp_dia_2"]].mean(axis=1)

        # ── Escanear carpetas de sujetos ─────────────────────────────────
        data_list = []
        try:
            all_entries = os.listdir(data_path)
        except FileNotFoundError:
            raise ValueError(f"Directorio no encontrado: {data_path}")

        subject_folders = sorted(
            [e for e in all_entries
             if os.path.isdir(os.path.join(data_path, e))
             and e.startswith("Sujeto")],
            key=lambda x: int(x.replace("Sujeto", ""))
        )

        if not subject_folders:
            raise ValueError(
                f"No se encontraron carpetas 'SujetoN' en: {data_path}"
            )

        skipped = []
        for folder_name in subject_folders:
            subject_id = folder_name          # e.g. "Sujeto1"
            folder_path = os.path.join(data_path, folder_name)
            video_filename = f"{subject_id}.mp4"
            video_path = os.path.join(folder_path, video_filename)

            # ── Verificar que el video existe ───────────────────────────
            if not os.path.isfile(video_path):
                print(f"[WARN] Video no encontrado: {video_path}")
                skipped.append(subject_id)
                continue

            # ── Obtener ground truth del CSV ─────────────────────────────
            if subject_id not in gt_df.index:
                print(f"[WARN] Sin ground truth para: {subject_id}")
                skipped.append(subject_id)
                continue

            row = gt_df.loc[subject_id]

            data_list.append({
                "index":          subject_id,
                "path":           folder_path,
                "video_filename": video_filename,
                "hr_gt":          float(row["hr_gt"]),
                "fr_gt":          float(row["fr_gt"]),
                "bp_sys":         float(row["bp_sys_mean"]),
                "bp_dia":         float(row["bp_dia_mean"]),
                "fitzpatrick":    int(row["fitzpatrick"]),
                "fp_group":       str(row["fitzpatrick_group"]),
                "gender":         str(row["gender"]),
                "age":            int(row["age"]),
            })

        if skipped:
            print(f"[INFO] Sujetos omitidos ({len(skipped)}): {skipped}")

        if not data_list:
            raise ValueError(
                f"No se encontraron sujetos válidos en: {data_path}"
            )

        print(f"[INFO] Sujetos válidos cargados: {len(data_list)}")
        self._print_fp_distribution(data_list)

        return data_list

    def split_raw_data(self, data_dirs, begin, end):
        """
        Divide la lista de sujetos en el rango porcentual [begin, end].
        Cada sujeto es una unidad independiente (un video por sujeto).
        """
        n = len(data_dirs)
        selected = data_dirs[int(n * begin):int(n * end)]
        print(f"[INFO] Split [{begin:.1f}-{end:.1f}]: {len(selected)} sujetos")
        return selected

    def preprocess_dataset(self, data_dirs, config_preprocess, begin, end):
        """
        Preprocesa los videos y guarda chunks en disco en el formato del toolbox.

        La señal BVP de ground truth se construye artificialmente a partir de la
        HR ground truth (onda sinusoidal a la frecuencia cardíaca medida), ya que
        el dataset ISB no incluye señal PPG de contacto continua.
        """
        data_dirs = self.split_raw_data(data_dirs, begin, end)

        skipped = []
        for item in tqdm(data_dirs, desc="Preprocesando ISB"):
            video_path = os.path.join(item["path"], item["video_filename"])

            try:
                frames = self.read_video(video_path)
                fps = config_preprocess.get("FS", 30)
                bvp = self._hr_to_synthetic_bvp(item["hr_gt"], len(frames), fps)
                frames_clips, bvps_clips = self.preprocess(frames, bvp, config_preprocess)
                self.save(frames_clips, bvps_clips, item["index"])
            except Exception as e:
                print(f"\n[SKIP] {item['index']}: {e}")
                skipped.append(item["index"])
                continue

        if skipped:
            print(f"\n[INFO] Sujetos saltados ({len(skipped)}): {skipped}")

        if not self.inputs:
            raise ValueError(
                "No se generaron archivos preprocesados. "
                "Verifica que los videos existan y sean legibles."
            )

        self.build_file_list({0: self.inputs})
        self.load_preprocessed_data()
        print(f"Dataset preprocesado listo: {self.preprocessed_data_len} chunks")

    # ------------------------------------------------------------------
    # Métodos de lectura estáticos
    # ------------------------------------------------------------------

    @staticmethod
    def read_video(video_path):
        """
        Lee un video MP4 y devuelve array NumPy RGB.
        Intenta OpenCV primero; fallback a imageio/ffmpeg.

        Returns:
            np.ndarray: shape (T, H, W, 3), dtype uint8
        """
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                return np.array(frames, dtype=np.uint8)

        try:
            import imageio
            reader = imageio.get_reader(video_path, format="ffmpeg")
            frames = [frame for frame in reader]
            reader.close()
            if frames:
                return np.array(frames, dtype=np.uint8)
        except Exception:
            pass

        raise IOError(
            f"No se pudo abrir el video: {video_path}\n"
            "Verifica que el archivo exista y que ffmpeg esté instalado."
        )

    @staticmethod
    def read_wave(dummy_path):
        """
        Placeholder — ISB no tiene señal PPG de contacto continua.
        La señal BVP se genera sintéticamente con _hr_to_synthetic_bvp().
        """
        raise NotImplementedError(
            "ISBLoader no lee señales PPG desde archivo. "
            "Usa _hr_to_synthetic_bvp() para generar BVP desde HR ground truth."
        )

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    @staticmethod
    def _hr_to_synthetic_bvp(hr_bpm, n_frames, fps=30):
        """
        Genera una señal BVP sinusoidal sintética a partir de la HR ground truth.

        Esta señal se usa únicamente para que el toolbox pueda calcular la HR
        estimada por cada método rPPG y compararla con hr_gt en el análisis
        posterior. No representa la morfología real del pulso.

        Args:
            hr_bpm  (float): frecuencia cardíaca en bpm
            n_frames (int) : número de frames del video
            fps      (int) : frames por segundo del video

        Returns:
            np.ndarray: señal BVP sintética 1D, dtype float32
        """
        t = np.linspace(0, n_frames / fps, n_frames)
        freq = hr_bpm / 60.0
        bvp = np.sin(2 * np.pi * freq * t).astype(np.float32)
        return bvp

    @staticmethod
    def get_video_info(video_path):
        """
        Inspecciona propiedades básicas de un video.

        Returns:
            dict: fps, total_frames, width, height, duration_sec
        """
        cap = cv2.VideoCapture(video_path)
        info = {
            "fps":          cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width":        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height":       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        if info["fps"] > 0:
            info["duration_sec"] = info["total_frames"] / info["fps"]
        cap.release()
        return info

    @staticmethod
    def _print_fp_distribution(data_list):
        """Imprime la distribución de grupos Fitzpatrick en consola."""
        from collections import Counter
        groups = Counter(item["fp_group"] for item in data_list)
        print("[INFO] Distribución Fitzpatrick:")
        for g in ["FP12", "FP3", "FP45"]:
            n = groups.get(g, 0)
            print(f"       {g}: {n} sujetos ({100*n/len(data_list):.1f}%)")


# =============================================================================
# Script de prueba rápida
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prueba rápida del ISBLoader")
    parser.add_argument("--data_path", required=True,
                        help="Ruta a ISB_DATASET/ (debe contener isb_ground_truth.csv)")
    parser.add_argument("--max_items", type=int, default=3,
                        help="Número de sujetos a inspeccionar")
    args = parser.parse_args()

    loader = ISBLoader.__new__(ISBLoader)
    data_list = loader.get_raw_data(args.data_path)

    print(f"\n{'='*60}")
    print(f"Total sujetos válidos: {len(data_list)}")
    print(f"{'='*60}")

    for item in data_list[:args.max_items]:
        info = ISBLoader.get_video_info(os.path.join(item["path"], item["video_filename"]))
        print(f"\nSujeto : {item['index']}")
        print(f"  FP   : {item['fitzpatrick']} ({item['fp_group']}) | "
              f"Género: {item['gender']} | Edad: {item['age']}")
        print(f"  HR   : {item['hr_gt']:.1f} bpm | FR: {item['fr_gt']:.1f} rpm")
        print(f"  BP   : {item['bp_sys']:.0f}/{item['bp_dia']:.0f} mmHg")
        print(f"  Video: {info['total_frames']} frames @ {info['fps']:.0f} fps "
              f"({info.get('duration_sec', 0):.1f}s) | "
              f"{info['width']}×{info['height']}px")
