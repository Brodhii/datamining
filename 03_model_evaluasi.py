"""
03_model_evaluasi.py  -  TAHAP 3: Model (Bab IV) dan Evaluasi (Bab V)

Alur:
  IV.2  Parameter analisis (ditetapkan di bagian PENGATURAN, bukan dari hasil)
  IV.3  Korelasi Spearman antar pasangan MK (n bersama dan rho dicatat)
  IV.4  Koreksi multiple comparisons (Benjamini-Hochberg FDR)
  IV.5  Graf: node = MK, edge = korelasi POSITIF lolos kriteria; negatif dipisah
  IV.6  Louvain (best-of-seed menurut modularity)
  V.1   Hasil korelasi dan pasangan terkuat
  V.2   Peta jaringan dan anggota komunitas
  V.3   Modularity, komponen terhubung, pembanding graf acak
  V.4   Analisis sensitivitas (n x rho x SD) dan stabilitas antar seed
  V.5   Dalam prodi vs lintas prodi (3 kategori: MK multi-prodi dipisahkan)
  V.6   Korelasi negatif (temuan tambahan)
  V.7   Pola paling kuat dan konsisten (bootstrap + konsistensi komunitas)

Masukan  : output/data/data_bersih.csv dan output/data/pemetaan_mk.csv (hasil 02_preprocessing.py)
Keluaran : output/tabel, output/gambar, output/data, output/catatan_angka_model.txt

Cara menjalankan:
    python 03_model_evaluasi.py                 (folder data default: output/data)
    python 03_model_evaluasi.py lokasi/data

Kebutuhan: pip install pandas numpy scipy networkx scikit-learn matplotlib
"""

import itertools
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# =====================================================================
# PENGATURAN: PARAMETER ANALISIS (IV.2)
# Nilai ini ditetapkan berdasarkan konvensi metodologis (lihat kolom alasan),
# BUKAN dipilih dari hasil. Jangan diubah setelah melihat hasil.
# =====================================================================
DIR_DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("output/data")
OUTPUT_DIR = Path("output")

N_MIN = 30        # jumlah minimum mahasiswa bersama per pasangan MK
RHO_MIN = 0.30    # kekuatan korelasi minimum untuk menjadi edge (jaringan utama)
RHO_KUAT = 0.50   # batas "korelasi kuat" (analisis dalam vs lintas prodi, pola kuat)
Q_FDR = 0.05      # batas false discovery rate (Benjamini-Hochberg)
SD_MIN = 1.0      # SD minimum nilai sebuah MK agar ikut dianalisis
RESOLUSI = 1.0    # resolusi Louvain (nilai baku)

N_SEED = 20       # jumlah seed Louvain (Louvain bersifat stokastik)
N_ACAK = 200      # jumlah graf acak pembanding modularity
N_BOOT = 200      # jumlah resampling bootstrap untuk kestabilan korelasi
SEED_UTAMA = 42

GRID_N = [20, 30, 40, 50]
GRID_RHO = [0.30, 0.40, 0.50]
GRID_SD_TAMBAHAN = [0.01, 2.0]   # 0.01 = hanya membuang MK yang nilainya konstan

BOOT_MIN_SHARE = 0.90   # kriteria "konsisten": share bootstrap dengan rho >= RHO_MIN
CO_MIN_SHARE = 0.80     # kriteria "konsisten": share skenario edge berada di komunitas yang sama
DPI = 200

# MK tugas akhir dan lapangan (Skripsi, Tesis, Seminar Proposal, KKL, PPL, dst.)
# False = tetap dianalisis (bagian dari struktur kurikulum) dan dibuatkan uji sensitivitas tanpa MK tsb.
# True  = dikeluarkan dari seluruh analisis.
KELUARKAN_MK_TUGAS_AKHIR = False
POLA_TUGAS_AKHIR = (r"^(skripsi|tesis|disertasi|tugas akhir( skripsi)?|seminar proposal|seminar hasil|"
                    r"kuliah kerja.*|kkn|kkl|kkp|ppl|pkl|praktik pengalaman lapangan.*|magang)$")

KAT_DALAM = "Dalam prodi (kedua MK satu prodi)"
KAT_MULTI = "Melibatkan MK multi-prodi"
KAT_LINTAS = "Lintas prodi (kedua MK satu prodi, prodi berbeda)"
WARNA_KAT = {KAT_DALAM: "#2f6fa8", KAT_MULTI: "#e08e0b", KAT_LINTAS: "#c0392b"}

ALASAN = {
    "Uji korelasi": "Spearman: hanya menuntut hubungan monoton dan lebih tahan outlier daripada Pearson; dua sisi.",
    "N_MIN": "Konvensi ukuran sampel minimum. Galat baku z-Fisher = 1/sqrt(n-3), sekitar 0,19 pada n = 30; "
             "pasangan yang lebih kecil terlalu tidak stabil dan menambah jumlah uji.",
    "RHO_MIN": "Batas bawah korelasi sedang menurut konvensi Cohen (1988).",
    "RHO_KUAT": "Batas korelasi kuat menurut konvensi Cohen (1988).",
    "Q_FDR": "Tingkat kesalahan penemuan palsu yang lazim. Benjamini-Hochberg dipilih karena Bonferroni "
             "terlalu konservatif untuk ratusan uji.",
    "SD_MIN": "MK dengan SD di bawah 1 poin hampir seragam; peringkatnya didominasi nilai kembar sehingga rho tidak stabil.",
    "RESOLUSI": "Nilai baku Louvain; tidak disetel agar tidak menyesuaikan hasil.",
    "N_SEED": "Louvain bergantung urutan acak; dijalankan berulang dan partisi bermodularity tertinggi dipakai.",
    "Korelasi negatif": "Tidak masuk jaringan utama; dilaporkan terpisah sebagai temuan tambahan.",
}

TABEL_DIR = OUTPUT_DIR / "tabel"
GAMBAR_DIR = OUTPUT_DIR / "gambar"
DATA_OUT = OUTPUT_DIR / "data"
for d in (TABEL_DIR, GAMBAR_DIR, DATA_OUT):
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BIRU, MERAH, ABU = "#2f6fa8", "#c0392b", "#9aa5b1"
rng = np.random.default_rng(SEED_UTAMA)
catatan = {}


# =====================================================================
# FUNGSI BANTU
# =====================================================================
def simpan_tabel(df, nama):
    path = TABEL_DIR / f"{nama}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"   [tabel]  {path}")


def simpan_gambar(fig, nama):
    path = GAMBAR_DIR / f"{nama}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    print(f"   [gambar] {path}")


def judul(teks):
    print("\n" + "=" * 70 + f"\n{teks}\n" + "=" * 70)


def bh_fdr(p):
    """Benjamini-Hochberg: kembalikan q-value."""
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return p
    urut = np.argsort(p)
    q = p[urut] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    hasil = np.empty(n)
    hasil[urut] = np.minimum(q, 1.0)
    return hasil


def louvain_terbaik(G, n_seed=N_SEED):
    """Jalankan Louvain dengan beberapa seed; kembalikan (komunitas, modularity, seed, semua_partisi)."""
    terbaik, semua = None, []
    for s in range(n_seed):
        kom = nx.community.louvain_communities(G, weight="weight", resolution=RESOLUSI, seed=s)
        q = nx.community.modularity(G, kom, weight="weight", resolution=RESOLUSI)
        semua.append(kom)
        if terbaik is None or q > terbaik[1] + 1e-12:
            terbaik = (kom, q, s)
    return terbaik[0], terbaik[1], terbaik[2], semua


def ke_label(kom, urutan_node):
    peta = {n: i for i, c in enumerate(kom) for n in c}
    return [peta[n] for n in urutan_node]


def partisi_dict(kom):
    """Beri nomor komunitas 1..K urut dari yang terbesar."""
    urut = sorted(kom, key=lambda c: (-len(c), min(c)))
    return {n: i + 1 for i, c in enumerate(urut) for n in c}


def bangun(T_semua, n_min, rho_min, sd_min):
    T = T_semua[(T_semua["n"] >= n_min) & (T_semua["sd_a"] >= sd_min) & (T_semua["sd_b"] >= sd_min)].copy()
    T["q"] = bh_fdr(T["p"].values)
    T["sig"] = T["q"] < Q_FDR
    E = T[T["sig"] & (T["rho"] >= rho_min)]
    Eneg = T[T["sig"] & (T["rho"] <= -rho_min)]
    G = nx.Graph()
    G.add_weighted_edges_from(zip(E["a"], E["b"], E["rho"]))
    return T, E, Eneg, G


def ari_nmi_bersama(p1, p2):
    """ARI dan NMI dua partisi (dict node->komunitas) pada node yang sama-sama ada."""
    bersama = sorted(set(p1) & set(p2))
    if len(bersama) < 2:
        return np.nan, np.nan, len(bersama)
    a = [p1[n] for n in bersama]
    b = [p2[n] for n in bersama]
    return adjusted_rand_score(a, b), normalized_mutual_info_score(a, b), len(bersama)


# =====================================================================
# 0. BACA DATA
# =====================================================================
judul("0. MEMBACA DATA HASIL PREPROCESSING")
f_data, f_peta = DIR_DATA / "data_bersih.csv", DIR_DATA / "pemetaan_mk.csv"
for f in (f_data, f_peta):
    if not f.exists():
        sys.exit(f"File {f} tidak ditemukan. Jalankan 02_preprocessing.py dulu.")
df = pd.read_csv(f_data, dtype={"NIM": str, "Kode MK": str})
peta = pd.read_csv(f_peta, dtype={"Kode MK": str, "Prodi utama": str}).set_index("Kode MK")
kode_ta = [k for k in peta.index if re.search(POLA_TUGAS_AKHIR, str(peta.loc[k, "Nama MK"]).strip(), re.I)]
print(f"MK tugas akhir/lapangan terdeteksi: {len(kode_ta)} kode "
      f"({'DIKELUARKAN dari analisis' if KELUARKAN_MK_TUGAS_AKHIR else 'tetap dianalisis; uji sensitivitas tanpa MK tsb disediakan'})")
if KELUARKAN_MK_TUGAS_AKHIR:
    df = df[~df["Kode MK"].isin(kode_ta)]
    peta = peta.drop(index=kode_ta, errors="ignore")
M = df.pivot(index="NIM", columns="Kode MK", values="Nilai Angka")
M_penuh = M.copy()   # seluruh MK sebelum penyaringan SD (untuk analisis cakupan)
print(f"Record {len(df)} | mahasiswa {M.shape[0]} | MK {M.shape[1]}")
catatan.update({"record": len(df), "mahasiswa": M.shape[0], "mk_awal": M.shape[1]})

sd_mk = M.std()
n_konstan = int((sd_mk.fillna(0) <= 0).sum())
print(f"MK dengan SD nol atau tak terhitung (tidak bisa dikorelasikan): {n_konstan}")
M = M[sd_mk[sd_mk > 0].index]
nama_mk = peta["Nama MK"]
prodi_mk = peta["Prodi utama"]
multi_mk = peta["MK multi-prodi"].astype(bool)

# Nama prodi (opsional): isi file prodi_nama.csv (kolom: Kode Prodi, Nama Prodi) agar kode prodi
# tampil sebagai nama di tabel dan gambar. Jika belum ada, skrip membuat templat untuk diisi.
f_prodi = DIR_DATA / "prodi_nama.csv"
NAMA_PRODI = {}
if f_prodi.exists():
    _pn = pd.read_csv(f_prodi, dtype=str).fillna("")
    NAMA_PRODI = {r["Kode Prodi"]: r["Nama Prodi"].strip() for _, r in _pn.iterrows() if r["Nama Prodi"].strip()}
    print(f"Nama prodi dibaca dari {f_prodi} ({len(NAMA_PRODI)} prodi)")
else:
    contoh_mk = (peta[~peta["MK multi-prodi"].astype(bool)].sort_values("Jumlah mahasiswa", ascending=False)
                 .groupby("Prodi utama")["Nama MK"].apply(lambda s: " | ".join(s.head(3))))
    templat = pd.DataFrame({"Kode Prodi": contoh_mk.index, "Nama Prodi": "", "Contoh MK di prodi ini": contoh_mk.values})
    templat.to_csv(DIR_DATA / "prodi_nama_TEMPLAT.csv", index=False, encoding="utf-8-sig")
    print(f"Templat nama prodi dibuat: {DIR_DATA / 'prodi_nama_TEMPLAT.csv'} "
          f"(isi kolom Nama Prodi, simpan sebagai prodi_nama.csv, lalu jalankan ulang)")


def label_prodi(kode):
    return f"{NAMA_PRODI[kode]} ({kode})" if kode in NAMA_PRODI else str(kode)


# =====================================================================
# IV.2 TABEL PARAMETER
# =====================================================================
judul("IV.2 PARAMETER ANALISIS")
param = pd.DataFrame([
    ["Uji korelasi", "Spearman, dua sisi", ALASAN["Uji korelasi"]],
    ["N_MIN", N_MIN, ALASAN["N_MIN"]],
    ["RHO_MIN", RHO_MIN, ALASAN["RHO_MIN"]],
    ["RHO_KUAT", RHO_KUAT, ALASAN["RHO_KUAT"]],
    ["Q_FDR", Q_FDR, ALASAN["Q_FDR"]],
    ["SD_MIN", SD_MIN, ALASAN["SD_MIN"]],
    ["RESOLUSI", RESOLUSI, ALASAN["RESOLUSI"]],
    ["N_SEED", N_SEED, ALASAN["N_SEED"]],
    ["Korelasi negatif", "Dilaporkan terpisah", ALASAN["Korelasi negatif"]],
], columns=["Parameter", "Nilai", "Alasan"])
print(param[["Parameter", "Nilai"]].to_string(index=False))
simpan_tabel(param, "tabel_4_1_parameter_analisis")


# =====================================================================
# IV.3 KORELASI SPEARMAN SEMUA PASANGAN (n >= batas terendah grid)
# =====================================================================
judul("IV.3 KORELASI SPEARMAN")
arr = M.values
kolom = list(M.columns)
ada = (~np.isnan(arr)).astype(int)
co = ada.T @ ada
batas_hitung = min(GRID_N)
pasangan = np.argwhere(np.triu(co >= batas_hitung, 1))
sd_all = M.std().values
prodi_mhs = (df.assign(**{"Kode Prodi": df["Kode Prodi"].astype(str)})
             .groupby("NIM")["Kode Prodi"].agg(lambda s: s.value_counts().index[0]).reindex(M.index).values)
baris, nan_rho = [], 0
for i, j in pasangan:
    m = ~np.isnan(arr[:, i]) & ~np.isnan(arr[:, j])
    x, y = arr[m, i], arr[m, j]
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        nan_rho += 1
        continue
    r, p = stats.spearmanr(x, y)
    if np.isnan(r):
        nan_rho += 1
        continue
    porsi = np.unique(prodi_mhs[m], return_counts=True)[1].max() / m.sum()
    baris.append((kolom[i], kolom[j], int(m.sum()), r, p, sd_all[i], sd_all[j], porsi))
P = pd.DataFrame(baris, columns=["a", "b", "n", "rho", "p", "sd_a", "sd_b", "porsi_prodi_mhs"])
P["prodi_a"] = P["a"].map(prodi_mk)
P["prodi_b"] = P["b"].map(prodi_mk)
# Pelabelan apa adanya menurut prodi utama (hanya untuk perbandingan literal; menyesatkan bila MK multi-prodi)
P["jenis"] = np.where(P["prodi_a"] == P["prodi_b"], "Dalam prodi", "Lintas prodi")
P["melibatkan_multi_prodi"] = P["a"].map(multi_mk) | P["b"].map(multi_mk)
# Kategori yang dipakai dalam analisis: MK multi-prodi (mis. MK umum, Skripsi) tidak bisa
# diberi satu prodi, jadi dipisahkan
P["kategori"] = np.select([P["melibatkan_multi_prodi"], P["prodi_a"] == P["prodi_b"]],
                          [KAT_MULTI, KAT_DALAM], default=KAT_LINTAS)
print(f"Pasangan dengan n >= {batas_hitung}: {len(pasangan)} | rho tak terdefinisi: {nan_rho} | dihitung: {len(P)}")
total_pasangan = len(kolom) * (len(kolom) - 1) // 2
catatan.update({"mk_dianalisis_SD_gt_0": len(kolom), "total_kemungkinan_pasangan": total_pasangan,
                f"pasangan_n_ge_{batas_hitung}": len(pasangan), "rho_tak_terdefinisi": nan_rho})


# =====================================================================
# IV.4 - IV.6 MODEL DASAR
# =====================================================================
judul("IV.4 - IV.6 MODEL DASAR")
T, E, Eneg, G = bangun(P, N_MIN, RHO_MIN, SD_MIN)
mk_uji = len(set(T["a"]) | set(T["b"]))
print(f"MK lolos SD >= {SD_MIN}: {int((pd.Series(sd_all) >= SD_MIN).sum())}")
print(f"Pasangan diuji (n >= {N_MIN}): {len(T)} | signifikan (q < {Q_FDR}): {int(T['sig'].sum())} "
      f"| positif: {int((T['sig'] & (T['rho'] > 0)).sum())} | negatif: {int((T['sig'] & (T['rho'] < 0)).sum())}")
print(f"Edge jaringan utama (positif, rho >= {RHO_MIN}): {len(E)} | node: {G.number_of_nodes()}")
if G.number_of_edges() == 0:
    sys.exit("Tidak ada edge yang lolos kriteria; periksa parameter.")

kom, Q_obs, seed_terbaik, semua_partisi = louvain_terbaik(G)
part = partisi_dict(kom)
urut_node = list(G.nodes())
labs = [ke_label(k, urut_node) for k in semua_partisi]
ari_seed = [adjusted_rand_score(labs[i], labs[j]) for i, j in itertools.combinations(range(len(labs)), 2)]
mod_seed = [nx.community.modularity(G, k, weight="weight", resolution=RESOLUSI) for k in semua_partisi]
komponen = [set(c) for c in nx.connected_components(G)]
kom_sama_komponen = sum(any(set(c) == k for k in komponen) for c in kom)
print(f"Louvain: {len(kom)} komunitas | modularity {Q_obs:.3f} | komponen terhubung {len(komponen)}")
print(f"Stabilitas {N_SEED} seed: rata-rata ARI {np.mean(ari_seed):.3f} (min {np.min(ari_seed):.3f}) "
      f"| modularity {np.min(mod_seed):.3f} - {np.max(mod_seed):.3f}")

catatan.update({
    "mk_diuji": mk_uji, "pasangan_diuji": len(T), "pasangan_signifikan": int(T["sig"].sum()),
    "signifikan_positif": int((T["sig"] & (T["rho"] > 0)).sum()),
    "signifikan_negatif": int((T["sig"] & (T["rho"] < 0)).sum()),
    "edge_jaringan_utama": len(E), "node_jaringan_utama": G.number_of_nodes(),
    "komunitas": len(kom), "modularity": round(Q_obs, 4), "komponen_terhubung": len(komponen),
    "komunitas_sama_dengan_komponen": kom_sama_komponen,
    "ari_antar_seed_rata2": round(float(np.mean(ari_seed)), 4),
})

# Tabel pasangan lengkap (lampiran)
T_out = T.copy()
T_out["nama_a"] = T_out["a"].map(nama_mk)
T_out["nama_b"] = T_out["b"].map(nama_mk)
T_out["status"] = np.select(
    [T_out["sig"] & (T_out["rho"] >= RHO_MIN), T_out["sig"] & (T_out["rho"] <= -RHO_MIN), T_out["sig"]],
    ["Edge positif (jaringan utama)", "Edge negatif (temuan tambahan)", "Signifikan, di bawah ambang rho"],
    default="Tidak signifikan")
simpan_tabel(T_out.round({"rho": 4, "p": 6, "q": 6}), "tabel_4_4_semua_pasangan_teruji")


# =====================================================================
# IV.3 CONTOH HITUNGAN MANUAL (bahan paragraf rumus di laporan)
# =====================================================================
judul("IV.3 CONTOH HITUNGAN MANUAL")
contoh_ok = False
for _, r in E.sort_values("rho", ascending=False).iterrows():
    sub = M[[r["a"], r["b"]]].dropna()
    pilih, pakai_x, pakai_y = [], set(), set()
    for nim, (x, y) in sub.iterrows():
        if round(x, 2) not in pakai_x and round(y, 2) not in pakai_y:
            pilih.append(nim)
            pakai_x.add(round(x, 2))
            pakai_y.add(round(y, 2))
        if len(pilih) == 8:
            break
    if len(pilih) == 8:
        contoh_ok = True
        break
if contoh_ok:
    s = sub.loc[pilih].copy()
    s.columns = ["Nilai_A", "Nilai_B"]
    s["Peringkat_A"] = s["Nilai_A"].rank()
    s["Peringkat_B"] = s["Nilai_B"].rank()
    s["d"] = s["Peringkat_A"] - s["Peringkat_B"]
    s["d_kuadrat"] = s["d"] ** 2
    nn = len(s)
    rho_manual = 1 - 6 * s["d_kuadrat"].sum() / (nn * (nn ** 2 - 1))
    print(f"Pasangan: {r['a']} ({nama_mk[r['a']]}) dan {r['b']} ({nama_mk[r['b']]})")
    print(s.round(2).to_string())
    print(f"sum d^2 = {s['d_kuadrat'].sum():.0f} | rho = 1 - 6*sum(d^2)/(n(n^2-1)) = {rho_manual:.4f} (n = {nn})")
    s_out = s.reset_index().round(2)
    s_out.loc[len(s_out)] = ["JUMLAH", None, None, None, None, None, s["d_kuadrat"].sum()]
    simpan_tabel(s_out, "tabel_4_2_contoh_hitung_manual_spearman")
    n_full, rho_full = int(r["n"]), float(r["rho"])
    t_stat = rho_full * np.sqrt((n_full - 2) / (1 - rho_full ** 2))
    p_manual = 2 * stats.t.sf(abs(t_stat), df=n_full - 2)
    uji_t = pd.DataFrame({
        "Keterangan": ["Pasangan", "n (seluruh mahasiswa bersama)", "rho", "t = rho*sqrt((n-2)/(1-rho^2))",
                       "df", "p (hitung manual)", "p (scipy.stats.spearmanr)"],
        "Nilai": [f"{r['a']} - {r['b']}", n_full, round(rho_full, 4), round(t_stat, 4), n_full - 2,
                  f"{p_manual:.3e}", f"{r['p']:.3e}"]})
    simpan_tabel(uji_t, "tabel_4_2b_contoh_uji_signifikansi")
    catatan["contoh_manual_pasangan"] = f"{r['a']} - {r['b']}"
else:
    print("Tidak ditemukan 8 mahasiswa dengan nilai berbeda pada pasangan edge; buat contoh manual sendiri.")


# =====================================================================
# V.1 HASIL KORELASI
# =====================================================================
judul("V.1 HASIL KORELASI")
top = T_out[T_out["sig"] & (T_out["rho"] > 0)].sort_values("rho", ascending=False).head(10)
top_out = top[["a", "nama_a", "b", "nama_b", "n", "rho", "q", "kategori"]].round({"rho": 3, "q": 6})
print(top_out[["a", "b", "n", "rho", "kategori"]].to_string(index=False))
simpan_tabel(top_out, "tabel_5_1_sepuluh_pasangan_terkuat")

fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-1, 1, 41)
ax.hist(T["rho"], bins=bins, color=ABU, edgecolor="white", label=f"Semua pasangan diuji ({len(T)})")
ax.hist(T.loc[T["sig"], "rho"], bins=bins, color=BIRU, edgecolor="white",
        label=f"Signifikan, q < {Q_FDR} ({int(T['sig'].sum())})")
for v in (-RHO_MIN, RHO_MIN):
    ax.axvline(v, color=MERAH, ls="--")
ax.set(title=f"Distribusi Korelasi Spearman antar Pasangan MK (n >= {N_MIN})",
       xlabel="Koefisien korelasi (rho)", ylabel="Jumlah pasangan")
ax.legend()
simpan_gambar(fig, "gambar_5_1_distribusi_rho")


# =====================================================================
# V.2 PETA JARINGAN DAN KOMUNITAS
# =====================================================================
judul("V.2 PETA JARINGAN DAN KOMUNITAS")
anggota = pd.DataFrame({
    "Komunitas": [part[n] for n in G.nodes()],
    "Kode MK": list(G.nodes()),
    "Nama MK": [nama_mk[n] for n in G.nodes()],
    "Prodi utama": [prodi_mk[n] for n in G.nodes()],
    "MK multi-prodi": [bool(multi_mk[n]) for n in G.nodes()],
    "Derajat": [G.degree(n) for n in G.nodes()],
    "Kekuatan (jumlah rho)": [round(G.degree(n, weight="weight"), 3) for n in G.nodes()],
}).sort_values(["Komunitas", "Derajat"], ascending=[True, False])
anggota.insert(4, "Nama prodi", anggota["Prodi utama"].map(label_prodi))
simpan_tabel(anggota, "tabel_5_2_anggota_komunitas")

ring = []
for c in sorted(set(part.values())):
    nodes = [n for n in part if part[n] == c]
    sub = G.subgraph(nodes)
    n_multi_k = sum(bool(multi_mk[n]) for n in nodes)
    pr = pd.Series([prodi_mk[n] for n in nodes if not multi_mk[n]] or [prodi_mk[n] for n in nodes])
    ring.append({
        "Komunitas": c, "Jumlah MK": len(nodes), "MK multi-prodi": n_multi_k, "Jumlah edge": sub.number_of_edges(),
        "Rata-rata rho": round(np.mean([d["weight"] for _, _, d in sub.edges(data=True)]), 3)
        if sub.number_of_edges() else np.nan,
        "Jumlah prodi (MK satu prodi)": pr.nunique(), "Prodi dominan": pr.value_counts().index[0],
        "Porsi prodi dominan": round(pr.value_counts(normalize=True).iloc[0], 2),
        "Sama dengan satu komponen": any(set(nodes) == k for k in komponen),
    })
ring = pd.DataFrame(ring)
ring.insert(6, "Nama prodi dominan", ring["Prodi dominan"].map(label_prodi))
print(ring.to_string(index=False))
simpan_tabel(ring, "tabel_5_3_ringkasan_komunitas")

# Tata letak: tiap komunitas di selnya sendiri agar terbaca
kelompok = {}
for n, c in part.items():
    kelompok.setdefault(c, []).append(n)
ncol = int(np.ceil(np.sqrt(len(kelompok))))
pos = {}
for k, (c, nodes) in enumerate(sorted(kelompok.items())):
    cx, cy = (k % ncol) * 4.2, -(k // ncol) * 4.0
    if len(nodes) == 1:
        p_loc = {nodes[0]: (0, 0)}
    elif len(nodes) == 2:
        p_loc = {nodes[0]: (-0.5, 0), nodes[1]: (0.5, 0)}
    else:
        p_loc = nx.spring_layout(G.subgraph(nodes), seed=SEED_UTAMA, scale=min(1.7, 0.55 + 0.11 * len(nodes)),
                                 k=2.2 / np.sqrt(len(nodes)), iterations=200)
    for n, (x, y) in p_loc.items():
        pos[n] = (cx + x, cy + y)
palet = plt.get_cmap("tab20")
warna = [palet((part[n] - 1) % 20) for n in G.nodes()]
lintas = [(u, v) for u, v in G.edges() if multi_mk[u] or multi_mk[v]]   # melibatkan MK multi-prodi
dalam = [(u, v) for u, v in G.edges() if not (multi_mk[u] or multi_mk[v])]

fig, ax = plt.subplots(figsize=(17, 12))
nx.draw_networkx_edges(G, pos, edgelist=dalam, ax=ax, edge_color="#8c8c8c", width=1.1)
nx.draw_networkx_edges(G, pos, edgelist=lintas, ax=ax, edge_color=WARNA_KAT[KAT_MULTI], width=1.8, style="dashed")
nx.draw_networkx_nodes(G, pos, ax=ax, node_color=warna, edgecolors="white",
                       node_size=[120 + 60 * G.degree(n) for n in G.nodes()])
nx.draw_networkx_labels(G, pos, ax=ax, font_size=5.5)
for c, nodes in kelompok.items():
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    ax.text(np.mean(xs), max(ys) + 0.28, f"Komunitas {c} ({len(nodes)} MK)", ha="center", fontsize=8, fontweight="bold")
ax.legend(handles=[plt.Line2D([0], [0], color="#8c8c8c", lw=1.5, label="Edge antar MK satu prodi"),
                   plt.Line2D([0], [0], color=WARNA_KAT[KAT_MULTI], lw=2, ls="--",
                              label="Edge yang melibatkan MK multi-prodi (mis. MK umum)")],
          loc="lower right")
ax.set_title(f"Peta Jaringan Korelasi MK (rho >= {RHO_MIN}, q < {Q_FDR}, n >= {N_MIN}); "
             f"warna = komunitas Louvain", fontweight="bold")
ax.axis("off")
simpan_gambar(fig, "gambar_5_2_peta_jaringan_komunitas")

# Ekspor untuk aplikasi (Bab VI)
E_out = E[["a", "b", "n", "rho", "q", "kategori", "porsi_prodi_mhs"]].copy()
E_out["komunitas_a"] = E_out["a"].map(part)
E_out["komunitas_b"] = E_out["b"].map(part)
E_out.to_csv(DATA_OUT / "edges_jaringan.csv", index=False, encoding="utf-8-sig")
anggota.to_csv(DATA_OUT / "nodes_komunitas.csv", index=False, encoding="utf-8-sig")
G_save = G.copy()
for n in G_save.nodes():
    G_save.nodes[n].update({"komunitas": int(part[n]), "nama": str(nama_mk[n]), "prodi": str(prodi_mk[n])})
nx.write_graphml(G_save, DATA_OUT / "jaringan_mk.graphml")
print(f"   [data]   {DATA_OUT / 'edges_jaringan.csv'}, nodes_komunitas.csv, jaringan_mk.graphml")


# =====================================================================
# V.2b PETA PER KOMUNITAS (label NAMA MK, tidak bertumpuk) - untuk laporan dan presentasi
# =====================================================================
judul("V.2b PETA PER KOMUNITAS (Nama MK)")


def bungkus(teks, lebar=16):
    return textwrap.fill(str(teks), lebar)


def tata_letak_inci(Hc, w, h):
    """Tata letak satu komunitas dengan satuan inci; titik digeser sampai kotak label (w x h) tidak tumpang tindih."""
    nodes = list(Hc.nodes())
    n = len(nodes)
    if n == 1:
        return {nodes[0]: np.zeros(2)}
    if n == 2:
        return {nodes[0]: np.array([-w * 0.75, 0.0]), nodes[1]: np.array([w * 0.75, 0.0])}
    try:
        p0 = nx.kamada_kawai_layout(Hc)
    except Exception:
        p0 = nx.spring_layout(Hc, seed=SEED_UTAMA)
    skala = max(2.0, 0.45 * np.sqrt(n) * w)
    pos = {k: np.array(v, float) * skala for k, v in p0.items()}
    for _ in range(800):
        gerak = False
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                d = pos[a] - pos[b]
                ox, oy = w - abs(d[0]), h - abs(d[1])
                if ox > 0 and oy > 0:
                    gerak = True
                    if ox / w < oy / h:
                        sg = 1.0 if d[0] >= 0 else -1.0
                        pos[a][0] += sg * (ox / 2 + 0.02)
                        pos[b][0] -= sg * (ox / 2 + 0.02)
                    else:
                        sg = 1.0 if d[1] >= 0 else -1.0
                        pos[a][1] += sg * (oy / 2 + 0.02)
                        pos[b][1] -= sg * (oy / 2 + 0.02)
        if not gerak:
            break
    return pos


def gambar_komunitas(c):
    nodes = [n for n in part if part[n] == c]
    H = G.subgraph(nodes)
    nama = {n: bungkus(nama_mk[n]) for n in nodes}
    maks_char = max(len(l) for t in nama.values() for l in t.split("\n"))
    maks_baris = max(t.count("\n") + 1 for t in nama.values())
    w = max(1.5, maks_char * 0.085 + 0.4)
    h = (maks_baris + 1) * 0.17 + 0.55
    pos = tata_letak_inci(H, w, h)
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x0, x1 = min(xs) - w / 2 - 0.3, max(xs) + w / 2 + 0.3
    y0, y1 = min(ys) - h - 0.1, max(ys) + 0.5
    fig, ax = plt.subplots(figsize=(max(6.0, x1 - x0), max(4.0, y1 - y0 + 0.7)))
    warna_k = plt.get_cmap("tab20")((c - 1) % 20)
    for u, v, dta in H.edges(data=True):
        multi = bool(multi_mk[u] or multi_mk[v])
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], zorder=1,
                color=WARNA_KAT[KAT_MULTI] if multi else "#8c8c8c", ls="--" if multi else "-",
                lw=1.0 + (dta["weight"] - RHO_MIN) * 9, alpha=0.75)
    for n in nodes:
        x, y = pos[n]
        ax.scatter([x], [y], s=150 + 70 * H.degree(n), color=warna_k, edgecolor="white", linewidth=1.5, zorder=3,
                   marker="D" if multi_mk[n] else "o")
        kotak = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8)
        ax.text(x, y - 0.17, nama[n], ha="center", va="top", fontsize=9, zorder=4, fontweight="bold", bbox=kotak)
        ax.text(x, y - 0.17 - (nama[n].count("\n") + 1) * 0.165 - 0.02, n, ha="center", va="top",
                fontsize=6.5, color="#666666", zorder=4, bbox=kotak)
    pr = pd.Series([prodi_mk[n] for n in nodes if not multi_mk[n]] or [prodi_mk[n] for n in nodes])
    ax.set_title(f"Komunitas {c}: {len(nodes)} mata kuliah | {H.number_of_edges()} keterkaitan | "
                 f"prodi dominan {label_prodi(pr.value_counts().index[0])}", fontsize=11, fontweight="bold")
    ax.legend(handles=[plt.Line2D([0], [0], color="#8c8c8c", lw=3, label="Keterkaitan (tebal garis = kekuatan rho)"),
                       plt.Line2D([0], [0], color=WARNA_KAT[KAT_MULTI], lw=2, ls="--", label="Melibatkan MK multi-prodi"),
                       plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=warna_k, markersize=8,
                                  label="MK multi-prodi (mis. MK umum)")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7, frameon=False)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    return fig


(GAMBAR_DIR / "komunitas").mkdir(exist_ok=True)
for c in sorted(set(part.values())):
    fig = gambar_komunitas(c)
    path = GAMBAR_DIR / "komunitas" / f"gambar_5_2b_komunitas_{c:02d}.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
print(f"   [gambar] {len(set(part.values()))} gambar per komunitas di {GAMBAR_DIR / 'komunitas'}")


# =====================================================================
# V.2c CAKUPAN MATA KULIAH: berapa MK yang ikut dan mengapa sebagian tidak
# =====================================================================
judul("V.2c CAKUPAN MATA KULIAH")
n_mhs_mk = M_penuh.notna().sum()
sd_penuh = M_penuh.std()
diuji_set = set(T["a"]) | set(T["b"])
sig_set = set(T.loc[T["sig"], "a"]) | set(T.loc[T["sig"], "b"])
net_set = set(G.nodes())
corong = pd.DataFrame([
    ["Kode MK setelah preprocessing", M_penuh.shape[1]],
    ["Nilainya bervariasi (SD > 0)", int((sd_penuh > 0).sum())],
    [f"SD >= {SD_MIN:g}", int((sd_penuh >= SD_MIN).sum())],
    [f"Diambil >= {N_MIN} mahasiswa", int((n_mhs_mk >= N_MIN).sum())],
    [f"Diambil >= {N_MIN} mahasiswa dan SD >= {SD_MIN:g}", int(((n_mhs_mk >= N_MIN) & (sd_penuh >= SD_MIN)).sum())],
    [f"Punya minimal 1 pasangan teruji (n >= {N_MIN})", len(diuji_set)],
    ["Punya minimal 1 pasangan signifikan (FDR)", len(sig_set)],
    ["Masuk jaringan utama (edge positif)", len(net_set)],
], columns=["Tahap penyaringan", "Jumlah MK"])
corong["Persen dari seluruh MK"] = (corong["Jumlah MK"] / M_penuh.shape[1] * 100).round(1)
print(corong.to_string(index=False))
simpan_tabel(corong, "tabel_5_9a_cakupan_mk_bertahap")

alasan = {f"Nilai terlalu seragam (SD < {SD_MIN:g})": 0, f"Diambil < {N_MIN} mahasiswa": 0,
          f"Tidak punya pasangan dengan n >= {N_MIN}": 0,
          "Diuji, tetapi tidak ada edge positif yang lolos (tidak signifikan, rho rendah, atau negatif)": 0}
for k in M_penuh.columns:
    if k in net_set:
        continue
    if not (sd_penuh.get(k, np.nan) >= SD_MIN):
        alasan[f"Nilai terlalu seragam (SD < {SD_MIN:g})"] += 1
    elif n_mhs_mk[k] < N_MIN:
        alasan[f"Diambil < {N_MIN} mahasiswa"] += 1
    elif k not in diuji_set:
        alasan[f"Tidak punya pasangan dengan n >= {N_MIN}"] += 1
    else:
        alasan["Diuji, tetapi tidak ada edge positif yang lolos (tidak signifikan, rho rendah, atau negatif)"] += 1
tab_alasan = pd.DataFrame({"Alasan MK tidak masuk jaringan": list(alasan), "Jumlah MK": list(alasan.values())})
tab_alasan["Persen dari MK yang tidak masuk"] = (tab_alasan["Jumlah MK"] / max(1, tab_alasan["Jumlah MK"].sum()) * 100).round(1)
print("\n" + tab_alasan.to_string(index=False))
simpan_tabel(tab_alasan, "tabel_5_9b_alasan_mk_tidak_masuk")

kunci_grup = prodi_mk.where(~multi_mk, "MULTI")
cov = pd.DataFrame({"grup": kunci_grup, "n_mhs": n_mhs_mk.reindex(kunci_grup.index),
                    "masuk": [k in net_set for k in kunci_grup.index]})
tab_cov = cov.groupby("grup").agg(**{"Jumlah MK": ("n_mhs", "size"), "Median mahasiswa per MK": ("n_mhs", "median"),
                                     f"MK diambil >= {N_MIN} mhs": ("n_mhs", lambda s: int((s >= N_MIN).sum())),
                                     "MK masuk jaringan": ("masuk", "sum")}).reset_index()
tab_cov["Persen masuk jaringan"] = (tab_cov["MK masuk jaringan"] / tab_cov["Jumlah MK"] * 100).round(0)
tab_cov.insert(1, "Nama prodi", tab_cov["grup"].map(lambda g: "MK multi-prodi (MK umum, Skripsi)" if g == "MULTI" else label_prodi(g)))
tab_cov = tab_cov.rename(columns={"grup": "Kode prodi"}).sort_values("Jumlah MK", ascending=False)
print("\n" + tab_cov.to_string(index=False))
simpan_tabel(tab_cov, "tabel_5_9c_cakupan_per_prodi")

fig, ax = plt.subplots(figsize=(9, 5))
tc = tab_cov.sort_values("Jumlah MK")
yy = np.arange(len(tc))
etiket = [f"{n}" if len(n) < 40 else n[:38] + "..." for n in tc["Nama prodi"]]
ax.barh(yy + 0.2, tc["Jumlah MK"], height=0.4, color=ABU, label="Seluruh MK")
ax.barh(yy - 0.2, tc["MK masuk jaringan"], height=0.4, color=BIRU, label="MK masuk jaringan")
for i, (a_, b_) in enumerate(zip(tc["Jumlah MK"], tc["MK masuk jaringan"])):
    ax.text(a_ + 0.3, i + 0.2, str(a_), va="center", fontsize=8)
    ax.text(b_ + 0.3, i - 0.2, str(int(b_)), va="center", fontsize=8)
ax.set_yticks(yy)
ax.set_yticklabels(etiket, fontsize=8)
ax.set(title="Cakupan Mata Kuliah dalam Jaringan per Prodi", xlabel="Jumlah mata kuliah")
ax.legend(loc="lower right")
simpan_gambar(fig, "gambar_5_7_cakupan_mk_per_prodi")
catatan.update({"mk_masuk_jaringan_persen": round(len(net_set) / M_penuh.shape[1] * 100, 1),
                "mk_diambil_ge_nmin_mhs": int((n_mhs_mk >= N_MIN).sum()),
                "mk_diambil_lt_nmin_mhs": int((n_mhs_mk < N_MIN).sum()),
                "prodi_tanpa_mk_di_jaringan": int((tab_cov.query("`Kode prodi` != 'MULTI'")["MK masuk jaringan"] == 0).sum())})


# =====================================================================
# V.3 MODULARITY DAN PEMBANDING ACAK
# =====================================================================
judul("V.3 MODULARITY DAN GRAF ACAK")
sizes_komponen = sorted([len(c) for c in komponen], reverse=True)
print(f"Komponen terhubung: {len(komponen)} (ukuran: {sizes_komponen[:10]}) "
      f"| komunitas yang identik dengan satu komponen: {kom_sama_komponen} dari {len(kom)}")

bobot_asli = np.array([d["weight"] for _, _, d in G.edges(data=True)])
Q_acak = []
for k in range(N_ACAK):
    Gr = nx.Graph(G)
    try:
        nx.double_edge_swap(Gr, nswap=10 * Gr.number_of_edges(), max_tries=200 * Gr.number_of_edges(),
                            seed=int(rng.integers(1e9)))
    except (nx.NetworkXError, nx.NetworkXAlgorithmError):
        pass
    bw = rng.permutation(bobot_asli)
    for (u, v), w in zip(Gr.edges(), bw):
        Gr[u][v]["weight"] = w
    _, qr, _, _ = louvain_terbaik(Gr, n_seed=3)
    Q_acak.append(qr)
Q_acak = np.array(Q_acak)
z = (Q_obs - Q_acak.mean()) / Q_acak.std() if Q_acak.std() > 0 else np.nan
p_acak = (np.sum(Q_acak >= Q_obs) + 1) / (N_ACAK + 1)
mod_tabel = pd.DataFrame({
    "Ukuran": ["Modularity jaringan nyata", "Rata-rata modularity graf acak", "SD modularity graf acak",
               "Modularity graf acak tertinggi", "z-score", "p empiris (satu sisi)",
               "Jumlah graf acak", "Jumlah komponen terhubung", "Komunitas identik dengan 1 komponen",
               "Jumlah komunitas", "Kepadatan graf"],
    "Nilai": [round(Q_obs, 4), round(Q_acak.mean(), 4), round(Q_acak.std(), 4), round(Q_acak.max(), 4),
              round(z, 2), round(p_acak, 4), N_ACAK, len(komponen), kom_sama_komponen, len(kom),
              round(nx.density(G), 4)]})
print(mod_tabel.to_string(index=False))
simpan_tabel(mod_tabel, "tabel_5_4_modularity_vs_graf_acak")
catatan.update({"modularity_acak_rata2": round(Q_acak.mean(), 4), "z_score_modularity": round(float(z), 2),
                "p_empiris_modularity": round(float(p_acak), 4)})

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(Q_acak, bins=25, color=ABU, edgecolor="white", label=f"Graf acak berderajat sama ({N_ACAK})")
ax.axvline(Q_obs, color=MERAH, lw=2, label=f"Jaringan nyata Q = {Q_obs:.3f}")
ax.set(title="Modularity Jaringan Nyata vs Graf Acak", xlabel="Modularity (Louvain)", ylabel="Jumlah graf")
ax.legend()
simpan_gambar(fig, "gambar_5_3_modularity_vs_acak")


# =====================================================================
# V.4 SENSITIVITAS
# =====================================================================
judul("V.4 ANALISIS SENSITIVITAS")
skenario = [(n, r, SD_MIN) for n in GRID_N for r in GRID_RHO]
skenario += [(N_MIN, RHO_MIN, s) for s in GRID_SD_TAMBAHAN]
baris_s, partisi_skenario = [], {}
for n_s, r_s, sd_s in skenario:
    T_s, E_s, _, G_s = bangun(P, n_s, r_s, sd_s)
    if G_s.number_of_edges() == 0:
        continue
    kom_s, q_s, _, _ = louvain_terbaik(G_s, n_seed=10)
    p_s = partisi_dict(kom_s)
    nama_sk = f"n>={n_s} | rho>={r_s:.2f} | SD>={sd_s:g}"
    partisi_skenario[nama_sk] = p_s
    ari_b, nmi_b, nb = ari_nmi_bersama(part, p_s)
    n_multi_s = int((E_s["kategori"] == KAT_MULTI).sum())
    n_lintas_s = int((E_s["kategori"] == KAT_LINTAS).sum())
    baris_s.append({
        "Skenario": nama_sk, "n min": n_s, "rho min": r_s, "SD min": sd_s,
        "Pasangan diuji": len(T_s), "Edge": len(E_s), "Node": G_s.number_of_nodes(),
        "Komunitas": len(kom_s), "Modularity": round(q_s, 3),
        "Komunitas terbesar": max(len(c) for c in kom_s),
        "Edge melibatkan MK multi-prodi": n_multi_s, "Edge lintas prodi murni": n_lintas_s,
        "ARI vs dasar": round(ari_b, 3) if pd.notna(ari_b) else np.nan,
        "NMI vs dasar": round(nmi_b, 3) if pd.notna(nmi_b) else np.nan,
        "Node bersama dgn dasar": nb,
        "Skenario dasar": (n_s, r_s, sd_s) == (N_MIN, RHO_MIN, SD_MIN),
    })
sens = pd.DataFrame(baris_s)
print(sens.drop(columns=["Skenario", "Skenario dasar"]).to_string(index=False))
simpan_tabel(sens, "tabel_5_5_sensitivitas")

# Heatmap grid n x rho (SD dasar)
grid = sens[sens["SD min"] == SD_MIN]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
for ax, (kol, jdl, cmap) in zip(axes, [("Komunitas", "Jumlah komunitas", "Blues"),
                                       ("Modularity", "Modularity", "Greens"),
                                       ("ARI vs dasar", "ARI terhadap skenario dasar", "Oranges")]):
    mat = grid.pivot(index="n min", columns="rho min", values=kol).reindex(index=GRID_N, columns=GRID_RHO)
    im = ax.imshow(mat.values.astype(float), cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(GRID_RHO)))
    ax.set_xticklabels([f"{r:.2f}" for r in GRID_RHO])
    ax.set_yticks(range(len(GRID_N)))
    ax.set_yticklabels(GRID_N)
    ax.grid(False)
    ax.set(title=jdl, xlabel="rho minimum", ylabel="n minimum")
    for a_i in range(len(GRID_N)):
        for b_i in range(len(GRID_RHO)):
            v = mat.values[a_i, b_i]
            if pd.notna(v):
                ax.text(b_i, a_i, f"{v:g}" if kol == "Komunitas" else f"{v:.2f}",
                        ha="center", va="center", fontsize=9)
fig.suptitle(f"Sensitivitas terhadap Ambang (SD min = {SD_MIN:g})", fontweight="bold")
simpan_gambar(fig, "gambar_5_4_sensitivitas_ambang")

dasar = sens[sens["Skenario dasar"]].iloc[0]
catatan["sensitivitas_ari_min"] = float(sens["ARI vs dasar"].min())
catatan["sensitivitas_ari_rata2"] = round(float(sens["ARI vs dasar"].mean()), 3)
catatan["sensitivitas_komunitas_min"] = int(sens["Komunitas"].min())

# --- Sensitivitas: tanpa MK tugas akhir / lapangan
kode_ta_ada = [k for k in kode_ta if k in set(M.columns)]
daftar_ta = pd.DataFrame({
    "Kode MK": kode_ta_ada,
    "Nama MK": [nama_mk[k] for k in kode_ta_ada],
    "Mahasiswa": [int(M[k].notna().sum()) for k in kode_ta_ada],
    "Rata-rata nilai": [round(M[k].mean(), 1) for k in kode_ta_ada],
    "SD nilai": [round(M[k].std(), 2) for k in kode_ta_ada],
    "MK multi-prodi": [bool(multi_mk[k]) for k in kode_ta_ada],
    "Masuk jaringan utama": [k in set(G.nodes()) for k in kode_ta_ada],
    "Jumlah edge": [int(((E["a"] == k) | (E["b"] == k)).sum()) for k in kode_ta_ada],
}).sort_values("Mahasiswa", ascending=False)
print(f"\nMK tugas akhir/lapangan pada data analisis: {len(daftar_ta)} | masuk jaringan: "
      f"{int(daftar_ta['Masuk jaringan utama'].sum())} | edge terkait: {int(daftar_ta['Jumlah edge'].sum())}")
simpan_tabel(daftar_ta, "tabel_5_5c_daftar_mk_tugas_akhir")
catatan["mk_tugas_akhir_pada_analisis"] = len(daftar_ta)
catatan["mk_tugas_akhir_masuk_jaringan"] = int(daftar_ta["Masuk jaringan utama"].sum())
catatan["edge_terkait_mk_tugas_akhir"] = int(daftar_ta["Jumlah edge"].sum())
if kode_ta_ada and not KELUARKAN_MK_TUGAS_AKHIR:
    P_ta = P[~P["a"].isin(kode_ta_ada) & ~P["b"].isin(kode_ta_ada)]
    T_t, E_t, _, G_t = bangun(P_ta, N_MIN, RHO_MIN, SD_MIN)
    kom_t, q_t, _, _ = louvain_terbaik(G_t, n_seed=10)
    p_t = partisi_dict(kom_t)
    ari_t, nmi_t, nb_t = ari_nmi_bersama(part, p_t)
    sens_ta = pd.DataFrame([
        {"Skenario": "Dasar (MK tugas akhir disertakan)", "Pasangan diuji": len(T), "Edge": len(E),
         "Node": G.number_of_nodes(), "Komunitas": len(kom), "Modularity": round(Q_obs, 3),
         "ARI vs dasar": 1.0, "NMI vs dasar": 1.0},
        {"Skenario": "Tanpa MK tugas akhir dan lapangan", "Pasangan diuji": len(T_t), "Edge": len(E_t),
         "Node": G_t.number_of_nodes(), "Komunitas": len(kom_t), "Modularity": round(q_t, 3),
         "ARI vs dasar": round(ari_t, 3), "NMI vs dasar": round(nmi_t, 3)}])
    print(sens_ta.to_string(index=False))
    simpan_tabel(sens_ta, "tabel_5_5b_sensitivitas_tanpa_mk_tugas_akhir")
    catatan.update({"tanpa_mk_tugas_akhir_edge": len(E_t), "tanpa_mk_tugas_akhir_node": G_t.number_of_nodes(),
                    "tanpa_mk_tugas_akhir_komunitas": len(kom_t), "tanpa_mk_tugas_akhir_ari": round(ari_t, 3)})
catatan["sensitivitas_komunitas_maks"] = int(sens["Komunitas"].max())


# =====================================================================
# V.5 DALAM PRODI VS LINTAS PRODI
# =====================================================================
judul("V.5 DALAM PRODI VS LINTAS PRODI")
# MK yang diselenggarakan banyak prodi (MK umum, Skripsi) tidak bisa diberi satu prodi, jadi
# pasangan dibagi tiga kategori. Sebagai pemeriksaan tambahan, diukur juga porsi mahasiswa dari
# prodi dominan pada tiap pasangan (apakah korelasi dibentuk oleh satu kelompok prodi).
is_edge = T["sig"] & (T["rho"] >= RHO_MIN)
is_kuat = T["sig"] & (T["rho"] >= RHO_KUAT)
kat_rows = []
for kat in (KAT_DALAM, KAT_MULTI, KAT_LINTAS):
    m = T["kategori"] == kat
    kat_rows.append({
        "Kategori pasangan MK": kat, "Pasangan diuji": int(m.sum()),
        "Edge (rho >= rho min)": int((m & is_edge).sum()),
        "Persen edge dari teruji": round((m & is_edge).sum() / m.sum() * 100, 2) if m.sum() else np.nan,
        "Edge kuat (rho >= rho kuat)": int((m & is_kuat).sum()),
        "Persen edge kuat dari teruji": round((m & is_kuat).sum() / m.sum() * 100, 2) if m.sum() else np.nan,
        "Rata-rata rho (edge)": round(T.loc[m & is_edge, "rho"].mean(), 3) if (m & is_edge).any() else np.nan,
        "Rata-rata rho (edge kuat)": round(T.loc[m & is_kuat, "rho"].mean(), 3) if (m & is_kuat).any() else np.nan,
        "Median rho (edge)": round(T.loc[m & is_edge, "rho"].median(), 3) if (m & is_edge).any() else np.nan,
        "Rata-rata rho (semua teruji)": round(T.loc[m, "rho"].mean(), 3) if m.any() else np.nan,
        "Rata-rata porsi mahasiswa prodi dominan": round(T.loc[m, "porsi_prodi_mhs"].mean(), 3) if m.any() else np.nan,
    })
kat_tab = pd.DataFrame(kat_rows)
md, mm = T["kategori"] == KAT_DALAM, T["kategori"] == KAT_MULTI
if md.sum() and mm.sum():
    p_fisher = stats.fisher_exact([[(md & is_edge).sum(), (md & ~is_edge).sum()],
                                   [(mm & is_edge).sum(), (mm & ~is_edge).sum()]])[1]
    p_mwu = stats.mannwhitneyu(T.loc[md, "rho"], T.loc[mm, "rho"])[1]
else:
    p_fisher = p_mwu = np.nan
kat_tab["p Fisher (dalam vs multi-prodi)"] = round(p_fisher, 4) if pd.notna(p_fisher) else np.nan
kat_tab["p Mann-Whitney rho (dalam vs multi-prodi)"] = round(p_mwu, 4) if pd.notna(p_mwu) else np.nan
print(kat_tab[["Kategori pasangan MK", "Pasangan diuji", "Edge (rho >= rho min)", "Persen edge dari teruji",
               "Edge kuat (rho >= rho kuat)", "Rata-rata rho (edge)"]].to_string(index=False))
simpan_tabel(kat_tab, "tabel_5_6_dalam_vs_lintas_prodi")

# Pelabelan literal berdasar prodi utama (untuk transparansi; lihat catatan di laporan)
lit = []
for jenis in ("Dalam prodi", "Lintas prodi"):
    m = T["jenis"] == jenis
    lit.append({"Pelabelan prodi utama": jenis, "Pasangan diuji": int(m.sum()),
                "Edge": int((m & is_edge).sum()), "Edge kuat": int((m & is_kuat).sum()),
                "Persen edge": round((m & is_edge).sum() / m.sum() * 100, 2) if m.sum() else np.nan,
                "Edge yang melibatkan MK multi-prodi": int((m & is_edge & T["melibatkan_multi_prodi"]).sum()),
                "Rata-rata rho (edge)": round(T.loc[m & is_edge, "rho"].mean(), 3) if (m & is_edge).any() else np.nan})
lit = pd.DataFrame(lit)
print("\nPelabelan literal berdasarkan prodi utama:")
print(lit.to_string(index=False))
simpan_tabel(lit, "tabel_5_6b_pelabelan_prodi_utama")

pur = T["porsi_prodi_mhs"]
pur_tab = pd.DataFrame({
    "Ukuran": ["Pasangan diuji", "Rata-rata porsi mahasiswa dari prodi dominan",
               "Median porsi", "Pasangan dengan porsi >= 0,8", "Pasangan dengan porsi >= 0,9",
               "Pasangan dengan porsi < 0,8 (campuran prodi)"],
    "Nilai": [len(T), round(pur.mean(), 3), round(pur.median(), 3), int((pur >= 0.8).sum()),
              int((pur >= 0.9).sum()), int((pur < 0.8).sum())]})
print("\n" + pur_tab.to_string(index=False))
simpan_tabel(pur_tab, "tabel_5_6c_porsi_mahasiswa_prodi_dominan")

# Komunitas vs prodi
node_list = list(G.nodes())
nmi_prodi = normalized_mutual_info_score([part[n] for n in node_list], [prodi_mk[n] for n in node_list])
ari_prodi = adjusted_rand_score([part[n] for n in node_list], [prodi_mk[n] for n in node_list])
tunggal = [n for n in node_list if not multi_mk[n]]
nmi_tunggal = normalized_mutual_info_score([part[n] for n in tunggal], [prodi_mk[n] for n in tunggal])
ari_tunggal = adjusted_rand_score([part[n] for n in tunggal], [prodi_mk[n] for n in tunggal])
print(f"\nKomunitas vs prodi utama (semua node): NMI {nmi_prodi:.3f} | ARI {ari_prodi:.3f}")
print(f"Komunitas vs prodi (hanya MK satu prodi, {len(tunggal)} node): NMI {nmi_tunggal:.3f} | ARI {ari_tunggal:.3f}")
catatan.update({
    "nmi_komunitas_vs_prodi_semua_node": round(nmi_prodi, 3), "ari_komunitas_vs_prodi_semua_node": round(ari_prodi, 3),
    "nmi_komunitas_vs_prodi_mk_satu_prodi": round(nmi_tunggal, 3),
    "ari_komunitas_vs_prodi_mk_satu_prodi": round(ari_tunggal, 3),
    "pasangan_diuji_dalam_prodi": int(md.sum()), "pasangan_diuji_melibatkan_multi_prodi": int(mm.sum()),
    "pasangan_diuji_lintas_prodi_murni": int((T["kategori"] == KAT_LINTAS).sum()),
    "edge_dalam_prodi": int((md & is_edge).sum()), "edge_melibatkan_multi_prodi": int((mm & is_edge).sum()),
    "edge_lintas_prodi_murni": int(((T["kategori"] == KAT_LINTAS) & is_edge).sum()),
    "porsi_mhs_prodi_dominan_rata2": round(float(pur.mean()), 3),
    "pasangan_porsi_mhs_kurang_0.8": int((pur < 0.8).sum()),
})

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
kt = kat_tab.copy()
singk = ["Dalam prodi\n(MK satu prodi)", "Melibatkan\nMK multi-prodi", "Lintas prodi\n(MK satu prodi)"]
nilai = kt["Persen edge dari teruji"].fillna(0)
ax[0].bar(singk, nilai, color=[WARNA_KAT[k] for k in kt["Kategori pasangan MK"]])
for i, (v, e_, t_) in enumerate(zip(nilai, kt["Edge (rho >= rho min)"], kt["Pasangan diuji"])):
    teks = f"{v:.1f}%\n({e_}/{t_})" if t_ else "tidak ada pasangan\nteruji (n >= batas)"
    ax[0].text(i, v, teks, ha="center", va="bottom", fontsize=8)
ax[0].set(title="Persentase pasangan teruji yang menjadi edge", ylabel="Persen")
ax[0].set_ylim(0, max(nilai.max() * 1.4, 5))
ax[1].hist(pur, bins=np.linspace(0, 1, 21), color=BIRU, edgecolor="white")
ax[1].axvline(0.8, color=MERAH, ls="--", label="0,8")
ax[1].set_yscale("log")
ax[1].set(title="Porsi mahasiswa dari prodi dominan\npada tiap pasangan teruji", xlabel="Porsi",
          ylabel="Jumlah pasangan (skala log)")
ax[1].legend()
fig.suptitle("Dalam Prodi vs Lintas Prodi", fontweight="bold")
simpan_gambar(fig, "gambar_5_5_dalam_vs_lintas_prodi")


# =====================================================================
# V.6 KORELASI NEGATIF
# =====================================================================
judul("V.6 KORELASI NEGATIF")
neg = T_out[T_out["sig"] & (T_out["rho"] < 0)].sort_values("rho")
neg_out = neg[["a", "nama_a", "b", "nama_b", "n", "rho", "q", "kategori"]].round({"rho": 3, "q": 6})
print(f"Pasangan negatif signifikan: {len(neg_out)} (rho <= -{RHO_MIN}: {len(Eneg)})")
print(neg_out.head(10)[["a", "b", "n", "rho", "kategori"]].to_string(index=False))
simpan_tabel(neg_out, "tabel_5_7_korelasi_negatif")
catatan["negatif_signifikan"] = len(neg_out)


# =====================================================================
# V.7 POLA PALING KUAT DAN KONSISTEN
# =====================================================================
judul("V.7 POLA KUAT DAN KONSISTEN")
bar = []
for _, r in E.iterrows():
    sub = M[[r["a"], r["b"]]].dropna()
    x, y = sub.iloc[:, 0].values, sub.iloc[:, 1].values
    n_ = len(sub)
    boot = []
    for _ in range(N_BOOT):
        ii = rng.integers(0, n_, n_)
        if np.ptp(x[ii]) == 0 or np.ptp(y[ii]) == 0:
            continue
        boot.append(stats.spearmanr(x[ii], y[ii])[0])
    boot = np.array(boot)
    share_boot = float(np.mean(boot >= RHO_MIN)) if len(boot) else np.nan
    ada_kom = [(p_s[r["a"]] == p_s[r["b"]]) for p_s in partisi_skenario.values()
               if r["a"] in p_s and r["b"] in p_s]
    bar.append({
        "Kode MK A": r["a"], "Nama MK A": nama_mk[r["a"]], "Kode MK B": r["b"], "Nama MK B": nama_mk[r["b"]],
        "n": n_, "rho": round(r["rho"], 3), "_rho": r["rho"], "q": r["q"], "Kategori": r["kategori"],
        "CI95 bawah": round(np.percentile(boot, 2.5), 3), "CI95 atas": round(np.percentile(boot, 97.5), 3),
        "Share bootstrap rho >= rho min": round(share_boot, 3),
        "Skenario dgn kedua MK": len(ada_kom),
        "Share satu komunitas": round(float(np.mean(ada_kom)), 3) if ada_kom else np.nan,
    })
pola = pd.DataFrame(bar)
pola["Kuat"] = pola["_rho"] >= RHO_KUAT      # bandingkan nilai TAK dibulatkan (0,4996 bukan "kuat")
pola = pola.drop(columns="_rho")
pola["Konsisten"] = (pola["Share bootstrap rho >= rho min"] >= BOOT_MIN_SHARE) & \
                    (pola["Share satu komunitas"] >= CO_MIN_SHARE)
pola["Kuat dan konsisten"] = pola["Kuat"] & pola["Konsisten"]
pola = pola.sort_values(["Kuat dan konsisten", "rho"], ascending=[False, False])
n_kk = int(pola["Kuat dan konsisten"].sum())
print(f"Edge jaringan utama: {len(pola)} | kuat (rho >= {RHO_KUAT}): {int(pola['Kuat'].sum())} "
      f"| konsisten: {int(pola['Konsisten'].sum())} | kuat dan konsisten: {n_kk}")
print(pola[pola["Kuat dan konsisten"]].head(10)[["Kode MK A", "Kode MK B", "n", "rho", "Kategori"]].to_string(index=False))
simpan_tabel(pola.round({"q": 6}), "tabel_5_8_pola_kuat_dan_konsisten")
catatan.update({"edge_kuat": int(pola["Kuat"].sum()), "edge_konsisten": int(pola["Konsisten"].sum()),
                "edge_kuat_dan_konsisten": n_kk})

tampil = pola[pola["Kuat dan konsisten"]].head(25).iloc[::-1]
if len(tampil):
    fig, ax = plt.subplots(figsize=(9, max(3, 0.28 * len(tampil) + 1.5)))
    yy = np.arange(len(tampil))
    warna_p = [WARNA_KAT[k] for k in tampil["Kategori"]]
    ax.hlines(yy, tampil["CI95 bawah"], tampil["CI95 atas"], color=warna_p, lw=2)
    ax.scatter(tampil["rho"], yy, c=warna_p, zorder=3)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{a} - {b}" for a, b in zip(tampil["Kode MK A"], tampil["Kode MK B"])], fontsize=7)
    ax.axvline(RHO_KUAT, color="black", ls=":", lw=1)
    ax.set(title="Pola Korelasi Paling Kuat dan Konsisten (rho dan CI 95% bootstrap)", xlabel="rho")
    ax.legend(handles=[plt.Line2D([0], [0], marker="o", color=WARNA_KAT[k], lw=2, label=k)
                       for k in (KAT_DALAM, KAT_MULTI) if k in set(tampil["Kategori"])], loc="lower right", fontsize=8)
    simpan_gambar(fig, "gambar_5_6_pola_kuat_konsisten")


# =====================================================================
# CATATAN ANGKA
# =====================================================================
judul("CATATAN ANGKA")
with open(OUTPUT_DIR / "catatan_angka_model.txt", "w", encoding="utf-8") as f:
    f.write(f"Parameter: N_MIN={N_MIN}, RHO_MIN={RHO_MIN}, RHO_KUAT={RHO_KUAT}, Q_FDR={Q_FDR}, SD_MIN={SD_MIN}, "
            f"RESOLUSI={RESOLUSI}, N_SEED={N_SEED}, N_ACAK={N_ACAK}, N_BOOT={N_BOOT}\n\n")
    for k, v in catatan.items():
        f.write(f"{k}: {v}\n")
        print(f"   {k}: {v}")
print(f"\n   [file]   {OUTPUT_DIR / 'catatan_angka_model.txt'}")
print("\nSELESAI. Hasil ada di folder:", OUTPUT_DIR.resolve())
