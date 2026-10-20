import os
import csv
import cv2
import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt

INPUT_DIR = "images"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DAFTAR_GAMBAR = [
    ("PLAT ASLI", "plat_asli.jpg"),
    ("PLAT TERANG", "plat_terang.jpg"),
    ("PLAT GELAP", "plat_gelap.jpg"),
]

WINDOW_W, WINDOW_H = 1400, 820
PANEL_W, PANEL_H = 680, 650
BG = (24, 25, 30)
PANEL = (36, 38, 45)
WHITE = (240, 242, 245)
MUTED = (165, 171, 182)
ACCENT = (70, 190, 255)
GREEN = (100, 220, 150)


def teks(img, text, xy, scale=0.6, color=WHITE, thickness=1):
    cv2.putText(img, str(text), xy, cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def pas_dalam(img, max_w, max_h):
    h, w = img.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((max_h, max_w, 3), dtype=np.uint8)
    skala = min(max_w / w, max_h / h)
    nw, nh = max(1, int(w * skala)), max(1, int(h * skala))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    kanvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)
    x, y = (max_w - nw) // 2, (max_h - nh) // 2
    kanvas[y:y + nh, x:x + nw] = resized
    return kanvas


def buat_panel_gambar(rgb, gray):
    panel = np.full((PANEL_H, PANEL_W, 3), PANEL, dtype=np.uint8)
    teks(panel, "IMAGE PREVIEW", (24, 34), 0.65, ACCENT, 2)
    teks(panel, "Gambar RGB", (24, 72), 0.55, WHITE, 1)
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    panel[88:323, 24:PANEL_W - 24] = pas_dalam(
        rgb_bgr, PANEL_W - 48, 235
    )
    teks(panel, "Hasil Grayscale", (24, 360), 0.55, WHITE, 1)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    panel[376:611, 24:PANEL_W - 24] = pas_dalam(
        gray_bgr, PANEL_W - 48, 235
    )
    return panel


def buat_histogram(gray):
    # OpenCV menghitung frekuensi setiap intensitas 0-255.
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist.astype(np.int64)

    chart = np.full((PANEL_H, PANEL_W, 3), PANEL, dtype=np.uint8)
    teks(chart, "GRAYSCALE HISTOGRAM", (24, 34), 0.65, ACCENT, 2)

    left, top = 48, 78
    plot_w, plot_h = PANEL_W - 80, 390
    right, bottom = left + plot_w, top + plot_h
    maksimum = max(1, int(hist.max()))

    for i in range(5):
        y = bottom - int(i * plot_h / 4)
        cv2.line(chart, (left, y), (right, y), (65, 68, 77), 1)
        teks(chart, f"{int(maksimum * i / 4):,}", (5, y + 5),
            0.36, MUTED, 1)

    for x in range(256):
        x1 = left + int(x * plot_w / 256)
        x2 = left + int((x + 1) * plot_w / 256)
        tinggi = int(hist[x] / maksimum * plot_h)
        cv2.rectangle(chart, (x1, bottom - tinggi),
                    (max(x1, x2 - 1), bottom), ACCENT, -1)

    cv2.rectangle(chart, (left, top), (right, bottom), (125, 130, 140), 1)
    for x in [0, 64, 128, 192, 255]:
        px = left + int(x / 255 * plot_w)
        teks(chart, str(x), (px - 10, bottom + 24), 0.42, MUTED, 1)

    teks(chart, "Intensitas piksel (0-255)", (left + 145, bottom + 50),
        0.5, WHITE, 1)
    teks(chart, "Jumlah Piksel", (24, 60), 0.42, MUTED, 1)

    total = int(gray.size)
    dominan = int(np.argmax(hist))
    rata = float(gray.mean())
    persen_gelap = float(np.count_nonzero(gray < 85) / total * 100)
    persen_terang = float(np.count_nonzero(gray > 170) / total * 100)

    teks(chart, f"Total piksel        : {total:,}", (28, 555), 0.52, WHITE, 1)
    teks(chart, f"Intensitas dominan  : {dominan} ({int(hist[dominan]):,} piksel)",
        (28, 583), 0.46, GREEN, 1)
    teks(chart, f"Rata-rata intensitas: {rata:.2f}", (28, 611), 0.48, WHITE, 1)
    teks(chart, f"Piksel gelap (<85): {persen_gelap:.1f}%", (28, 639),
        0.43, MUTED, 1)
    teks(chart, f"Piksel terang (>170): {persen_terang:.1f}%", (350, 639),
        0.43, MUTED, 1)
    return chart, hist


def simpan_hasil(nama, gray, hist):
    dasar = nama.lower().replace(" ", "_")
    iio.imwrite(os.path.join(OUTPUT_DIR, f"{dasar}_grayscale.png"), gray)

    with open(os.path.join(OUTPUT_DIR, f"{dasar}_histogram.csv"),
            "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Intensitas", "Jumlah Piksel"])
        writer.writerows((i, int(hist[i])) for i in range(256))

    # Matplotlib menyimpan grafik histogram untuk kebutuhan laporan.
    plt.figure(figsize=(10, 4.8))
    plt.bar(np.arange(256), hist, width=1.0, color="gray")
    plt.title(f"Histogram Grayscale - {nama}")
    plt.xlabel("Intensitas Piksel (0-255)")
    plt.ylabel("Jumlah Piksel")
    plt.xlim(0, 255)
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{dasar}_histogram.png"), dpi=300)
    plt.close()


def muat_gambar(path):
    rgb = iio.imread(path)
    if rgb.ndim == 2:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
    elif rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Format gambar harus RGB, RGBA, atau grayscale.")

    if rgb.dtype != np.uint8:
        arr = rgb.astype(np.float64)
        mn, mx = arr.min(), arr.max()
        arr = (arr - mn) * 255.0 / (mx - mn) if mx > mn else np.zeros_like(arr)
        rgb = np.clip(arr, 0, 255).astype(np.uint8)

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return rgb, gray


def main():
    tersedia = []
    for nama, file in DAFTAR_GAMBAR:
        path = os.path.join(INPUT_DIR, file)
        if os.path.isfile(path):
            tersedia.append((nama, path))
        else:
            print(f"[PERINGATAN] Gambar tidak ditemukan: {path}")

    if not tersedia:
        print("Tidak ada gambar ditemukan. Pastikan file ada di folder images.")
        return

    data = []
    for nama, path in tersedia:
        try:
            rgb, gray = muat_gambar(path)
            hist = np.bincount(gray.ravel(), minlength=256)
            simpan_hasil(nama, gray, hist)
            data.append((nama, rgb, gray, hist))
            print(f"{nama}: ukuran={gray.shape}, total piksel={gray.size:,}, "
                f"intensitas dominan={int(np.argmax(hist))}, "
                f"frekuensi={int(hist.max()):,}")
        except Exception as e:
            print(f"[ERROR] Gagal memproses {path}: {e}")

    if not data:
        return

    index = 0
    window_name = "Analisis Histogram Citra - OpenCV"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, WINDOW_W, WINDOW_H)

    while True:
        nama, rgb, gray, hist_np = data[index]
        canvas = np.full((WINDOW_H, WINDOW_W, 3), BG, dtype=np.uint8)
        teks(canvas, "ANALISIS HISTOGRAM CITRA", (28, 38), 0.9, WHITE, 2)
        teks(canvas, f"Objek: {nama}", (30, 72), 0.58, ACCENT, 2)
        teks(canvas, "OpenCV | NumPy | ImageIO | Matplotlib",
            (WINDOW_W - 530, 38), 0.48, MUTED, 1)

        panel_gambar = buat_panel_gambar(rgb, gray)
        panel_hist, _ = buat_histogram(gray)
        y_start = 105
        canvas[y_start:y_start + PANEL_H, 20:20 + PANEL_W] = panel_gambar
        canvas[y_start:y_start + PANEL_H, 700:700 + PANEL_W] = panel_hist

        teks(canvas, "Pilih gambar: [1] Asli  [2] Terang  [3] Gelap",
            (25, 790), 0.52, WHITE, 1)
        teks(canvas, "[A] Simpan tampilan  |  [Q] Keluar",
            (650, 790), 0.52, GREEN, 1)
        cv2.imshow(window_name, canvas)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        elif key == ord("1") and len(data) >= 1:
            index = 0
        elif key == ord("2") and len(data) >= 2:
            index = 1
        elif key == ord("3") and len(data) >= 3:
            index = 2
        elif key in (ord("a"), ord("A")):
            dasar = nama.lower().replace(" ", "_")
            path = os.path.join(OUTPUT_DIR, f"{dasar}_ui_opencv.png")
            cv2.imwrite(path, canvas)
            print(f"Tampilan disimpan: {path}")

    cv2.destroyAllWindows()
    print(f"Selesai. Semua hasil tersimpan di folder '{OUTPUT_DIR}'.")


if __name__ == "__main__":
    main()
