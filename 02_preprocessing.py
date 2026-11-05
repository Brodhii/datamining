"""
02_preprocessing.py  -  TAHAP 2: Preprocessing (Bab III laporan)

Langkah yang dikerjakan (tiap langkah dicatat di tabel alur jumlah data):
  III.1  Anonimisasi NIM & nama     -> sudah dilakukan di skrip terpisah (ubah_data.py), dicatat saja
  III.2  Perbaikan nilai tidak valid -> typo seperti "91..00" menjadi 91.00
  III.3  Penanganan missing value    -> Nilai Angka kosong / teks tak bermakna dikeluarkan
  III.4  Penanganan Nilai 0 dan E    -> sesuai MODE_E (lihat pengaturan)
  III.5  Standarisasi Nama MK        -> spasi, newline, tanda asterisk, variasi penulisan
  III.6  Penentuan prodi acuan       -> prodi utama tiap MK + penanda MK multi-prodi
  III.7  Validasi akhir              -> duplikat dan konsistensi Nilai Huruf vs Nilai Angka
  III.8  Transformasi                -> matriks mahasiswa x MK (bahan Tahap 3)

Catatan desain:
  * Node jaringan = KODE MK (bukan Nama MK). Nama yang sama dengan kode berbeda adalah
    MK berbeda milik prodi berbeda (mis. "Fikih Muamalah" di prodi 62202 dan 86230),
    jadi TIDAK digabung. Menggabungkan berdasarkan nama akan membuat hubungan lintas
    prodi palsu.
  * Ambang n minimum dan SD minimum BELUM diterapkan di sini. Keduanya adalah parameter
    analisis (Bab IV.2) yang diterapkan di Tahap 3.

Cara menjalankan:
    python 02_preprocessing.py                  (file data_anonim.xlsx di folder yang sama)
    python 02_preprocessing.py nama_file.xlsx

Kebutuhan: pip install pandas openpyxl matplotlib numpy
"""

import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# =====================================================================
# PENGATURAN
# =====================================================================
FILE_INPUT = sys.argv[1] if len(sys.argv) > 1 else "data_anonim.xlsx"
OUTPUT_DIR = Path("output")
PRODI_COL = "Kode Prodi Kelas"   # prodi acuan MK (sama dengan Tahap 1)
DPI = 200

# Perlakuan huruf E:
#   "tanpa_capaian" : keluarkan record yang tidak punya capaian terukur
#                     (Nilai Angka kosong, teks tak bermakna, atau 0).
#                     E dengan Nilai Angka > 0 dipertahankan sebagai capaian rendah yang valid.
#   "semua_E"       : keluarkan seluruh record berhuruf E.
MODE_E = "tanpa_capaian"

# MK tugas akhir dan lapangan (bukan kelas reguler). Data TIDAK dihapus di tahap ini; hanya didata
# dalam tabel_3_4b. Perlakuannya dalam analisis (disertakan / dikeluarkan) diatur di 03_model_evaluasi.py.
POLA_TUGAS_AKHIR = (r"^(skripsi|tesis|disertasi|tugas akhir( skripsi)?|seminar proposal|seminar hasil|"
                    r"kuliah kerja.*|kkn|kkl|kkp|ppl|pkl|praktik pengalaman lapangan.*|magang)$")

# Batas bawah Nilai Angka untuk tiap Nilai Huruf (dipakai validasi konsistensi)
BATAS_HURUF = [("A", 90), ("A-", 85), ("B+", 80), ("B", 75), ("B-", 70),
               ("C+", 65), ("C", 60), ("C-", 55), ("D", 50), ("E", -np.inf)]

TABEL_DIR = OUTPUT_DIR / "tabel"
GAMBAR_DIR = OUTPUT_DIR / "gambar"
DATA_DIR = OUTPUT_DIR / "data"
for d in (TABEL_DIR, GAMBAR_DIR, DATA_DIR):
    d.mkdir(parents=True, exist_ok=True)

_MPL = tuple(int(v) for v in matplotlib.__version__.split(".")[:2])
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})
BIRU, MERAH = "#2f6fa8", "#c0392b"

if MODE_E not in ("tanpa_capaian", "semua_E"):
    sys.exit("MODE_E harus 'tanpa_capaian' atau 'semua_E'")


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


def huruf_dari_angka(a):
    if pd.isna(a):
        return np.nan
    for h, batas in BATAS_HURUF:
        if a >= batas:
            return h


alur = []  # catatan alur jumlah data


def catat(langkah, keterangan, d, dikeluarkan=0):
    alur.append({
        "Langkah": langkah,
        "Keterangan": keterangan,
        "Record dikeluarkan": dikeluarkan,
        "Record tersisa": len(d),
        "Mahasiswa tersisa": d["NIM"].nunique(),
        "Kode MK tersisa": d["Kode MK"].nunique(),
    })
    print(f"   {langkah:<7}{keterangan}")
    print(f"          dikeluarkan={dikeluarkan:>5} | record={len(d):>6} | "
          f"mahasiswa={d['NIM'].nunique():>5} | MK={d['Kode MK'].nunique():>4}")


# =====================================================================
# 0. BACA DATA MENTAH (ANONIM)
# =====================================================================
judul("0. MEMBACA DATA")
if not Path(FILE_INPUT).exists():
    sys.exit(f"File '{FILE_INPUT}' tidak ditemukan.")
raw = pd.read_excel(FILE_INPUT, dtype={"NIM": str, "Kode MK": str})
raw["Kode Prodi"] = raw["Kode Prodi"].astype(str)
raw[PRODI_COL] = raw[PRODI_COL].astype(str)
df = raw.copy()
angka_sebelum = pd.to_numeric(raw["Nilai Angka"], errors="coerce")
print(f"Mode perlakuan huruf E: {MODE_E}")
catat("Awal", "Data mentah (NIM dan nama sudah dianonimkan)", df)

contoh = []   # contoh sebelum/sesudah untuk laporan
catatan = {"mode_E": MODE_E, "record_awal": len(df), "mahasiswa_awal": df["NIM"].nunique(),
           "kode_mk_awal": df["Kode MK"].nunique()}


# =====================================================================
# III.2 PERBAIKAN NILAI TIDAK VALID (typo)
# =====================================================================
judul("III.2 PERBAIKAN NILAI TIDAK VALID")
df["_raw_angka"] = df["Nilai Angka"]
angka = pd.to_numeric(df["Nilai Angka"], errors="coerce")
mask_teks = angka.isna() & df["Nilai Angka"].notna()
diperbaiki = 0
for idx in df.index[mask_teks]:
    asli = str(df.at[idx, "Nilai Angka"]).strip()
    kandidat = re.sub(r"\.{2,}", ".", asli)           # "91..00" -> "91.00"
    nilai = pd.to_numeric(kandidat, errors="coerce")
    if pd.notna(nilai) and 0 <= nilai <= 100 and huruf_dari_angka(nilai) == df.at[idx, "Nilai Huruf"]:
        df.at[idx, "Nilai Angka"] = nilai             # perbaikan hanya jika cocok dengan Nilai Huruf
        diperbaiki += 1
        contoh.append(["Typo Nilai Angka", "Nilai Angka", asli, f"{nilai:.2f}"])
        print(f"   diperbaiki: {asli!r} -> {nilai} (Nilai Huruf {df.at[idx, 'Nilai Huruf']})")
df["Nilai Angka"] = pd.to_numeric(df["Nilai Angka"], errors="coerce")
catat("III.2", f"Perbaikan typo Nilai Angka ({diperbaiki} record diperbaiki)", df)
catatan["nilai_angka_diperbaiki"] = diperbaiki


# =====================================================================
# III.3 MISSING VALUE / TEKS TAK BERMAKNA
# =====================================================================
judul("III.3 PENANGANAN MISSING VALUE")
tak_valid = df["Nilai Angka"].isna()
n_kosong = int(df.loc[tak_valid, "_raw_angka"].isna().sum())
n_teks = int(tak_valid.sum() - n_kosong)
print(f"   Nilai Huruf pada record tanpa angka: {df.loc[tak_valid, 'Nilai Huruf'].value_counts().to_dict()}")
# Rincian MK untuk dasar argumentasi (disimpan sebagai tabel)
rincian = (df[tak_valid].groupby(["Kode MK", "Nama MK"]).size()
           .sort_values(ascending=False).reset_index(name="Record tanpa Nilai Angka"))
simpan_tabel(rincian.head(15), "tabel_3_3_mk_dengan_nilai_angka_kosong")
d_sebelum = len(df)
df = df[~tak_valid].copy()
catat("III.3", f"Keluarkan Nilai Angka kosong ({n_kosong}) dan teks tak bermakna ({n_teks})",
      df, d_sebelum - len(df))
catatan["dikeluarkan_kosong"] = n_kosong
catatan["dikeluarkan_teks"] = n_teks


# =====================================================================
# III.4 NILAI 0 DAN HURUF E
# =====================================================================
judul("III.4 PENANGANAN NILAI 0 DAN HURUF E")
E = raw[raw["Nilai Huruf"] == "E"]
tot_per_mhs = raw.groupby("NIM").size()
e_per_mhs = E.groupby("NIM").size()
semua_E = e_per_mhs[e_per_mhs == tot_per_mhs.reindex(e_per_mhs.index)]
print(f"   Record E: {len(E)} | mahasiswa dengan E: {len(e_per_mhs)} | "
      f"mahasiswa yang SEMUA record-nya E: {len(semua_E)} ({int(tot_per_mhs.reindex(semua_E.index).sum())} record)")
top_e = E.groupby("Nama MK").size().sort_values(ascending=False).head(5)
print("   Nama MK dengan E terbanyak:", top_e.to_dict())
bukti_e = pd.DataFrame({
    "Ukuran": ["Record berhuruf E", "Mahasiswa yang memiliki E",
               "Mahasiswa yang semua record-nya E", "Record milik mahasiswa tsb",
               "Rata-rata Nilai Indeks pada record E"],
    "Nilai": [len(E), len(e_per_mhs), len(semua_E),
              int(tot_per_mhs.reindex(semua_E.index).sum()),
              round(pd.to_numeric(E["Nilai Indeks"], errors="coerce").mean(), 2)],
})
simpan_tabel(bukti_e, "tabel_3_4_bukti_huruf_E")

d_sebelum = len(df)
mask_nol = df["Nilai Angka"] == 0
df = df[~mask_nol].copy()
catat("III.4a", "Keluarkan Nilai Angka = 0 (tidak ada capaian terukur)", df, d_sebelum - len(df))
catatan["dikeluarkan_nol"] = int(d_sebelum - len(df))

if MODE_E == "semua_E":
    d_sebelum = len(df)
    df = df[df["Nilai Huruf"] != "E"].copy()
    catat("III.4b", "Keluarkan sisa record berhuruf E (Nilai Angka > 0)", df, d_sebelum - len(df))
    catatan["dikeluarkan_E_sisa"] = int(d_sebelum - len(df))
else:
    sisa_e = int((df["Nilai Huruf"] == "E").sum())
    print(f"   E dengan Nilai Angka > 0 dipertahankan: {sisa_e} record")
    catatan["E_dipertahankan"] = sisa_e


# =====================================================================
# III.5 STANDARISASI NAMA MK
# =====================================================================
judul("III.5 STANDARISASI NAMA MK")
df["Kode MK"] = df["Kode MK"].str.strip()
nama_asli = df["Nama MK"].astype(str)
df["ada_asterisk"] = nama_asli.str.contains(r"\*", regex=True)
bersih = (nama_asli.str.replace("*", "", regex=False)
          .str.replace(r"\s+", " ", regex=True).str.strip())    # spasi ganda, newline, spasi ujung
df["_nama_bersih"] = bersih

n_berubah = int((bersih != nama_asli).sum())
for asli, baru in zip(nama_asli[bersih != nama_asli].drop_duplicates().head(4),
                      bersih[bersih != nama_asli].drop_duplicates().head(4)):
    contoh.append(["Pembersihan Nama MK", "Nama MK", repr(asli), repr(baru)])

# Satu nama baku per Kode MK: varian paling sering (abaikan huruf besar/kecil),
# lalu pilih penulisan dengan kapitalisasi terbaik.
def nama_baku(s):
    kunci = s.str.casefold()
    kunci_utama = kunci.value_counts().index[0]
    varian = s[kunci == kunci_utama]
    return sorted(varian.unique(), key=lambda v: (-sum(c.isupper() for c in v), v))[0]

baku = df.groupby("Kode MK")["_nama_bersih"].apply(nama_baku)
variasi = df.groupby("Kode MK")["_nama_bersih"].nunique()
for kode in variasi[variasi > 1].index:
    varian = df.loc[df["Kode MK"] == kode, "_nama_bersih"].value_counts().to_dict()
    contoh.append(["Variasi penulisan Nama MK", f"Kode MK {kode}", str(varian), baku[kode]])
    print(f"   {kode}: {varian} -> {baku[kode]!r}")

nama_unik_sebelum = raw["Nama MK"].nunique()
df["Nama MK"] = df["Kode MK"].map(baku)
catat("III.5", f"Standarisasi Nama MK ({n_berubah} record dibersihkan; "
               f"nama unik {nama_unik_sebelum} -> {df['Nama MK'].nunique()})", df)
catatan["record_nama_dibersihkan"] = n_berubah
catatan["nama_mk_unik_sebelum"] = int(nama_unik_sebelum)
catatan["nama_mk_unik_sesudah"] = int(df["Nama MK"].nunique())
catatan["kode_mk_dengan_>1_nama_sesudah"] = int((df.groupby("Kode MK")["Nama MK"].nunique() > 1).sum())

# Pendataan MK tugas akhir / lapangan (tidak dihapus)
nama_raw_bersih = (raw["Nama MK"].astype(str).str.replace("*", "", regex=False)
                   .str.replace(r"\s+", " ", regex=True).str.strip())
m_ta = nama_raw_bersih.str.contains(POLA_TUGAS_AKHIR, case=False, regex=True)
if m_ta.any():
    kode_raw = raw["Kode MK"].astype(str).str.strip()
    ta = pd.DataFrame({"Kode MK": kode_raw[m_ta], "Nama MK": nama_raw_bersih[m_ta]})
    tab_ta = ta.groupby("Kode MK").agg(**{"Nama MK": ("Nama MK", lambda x: x.value_counts().index[0]),
                                          "Record mentah": ("Nama MK", "size")}).reset_index()
    tab_ta["Record valid (dipertahankan)"] = tab_ta["Kode MK"].map(df.groupby("Kode MK").size()).fillna(0).astype(int)
    tab_ta["Dikeluarkan (tanpa capaian)"] = tab_ta["Record mentah"] - tab_ta["Record valid (dipertahankan)"]
    tab_ta["Mahasiswa valid"] = tab_ta["Kode MK"].map(df.groupby("Kode MK")["NIM"].nunique()).fillna(0).astype(int)
    tab_ta = tab_ta.sort_values("Record mentah", ascending=False)
    print(f"   MK tugas akhir/lapangan: {len(tab_ta)} kode | record mentah {int(tab_ta['Record mentah'].sum())} "
          f"| valid {int(tab_ta['Record valid (dipertahankan)'].sum())} (TIDAK dihapus dari data)")
    simpan_tabel(tab_ta, "tabel_3_4b_mk_tugas_akhir_dan_lapangan")
    catatan["mk_tugas_akhir_kode"] = len(tab_ta)
    catatan["mk_tugas_akhir_record_mentah"] = int(tab_ta["Record mentah"].sum())
    catatan["mk_tugas_akhir_record_valid"] = int(tab_ta["Record valid (dipertahankan)"].sum())

# Nama sama tetapi kode berbeda: DIPERTAHANKAN sebagai MK terpisah (dokumentasikan)
nama_multi = df.groupby("Nama MK")["Kode MK"].nunique()
nama_multi = nama_multi[nama_multi > 1]
tabel_nama_multi = pd.DataFrame([
    {"Nama MK": n,
     "Jumlah kode": int(nama_multi[n]),
     "Daftar kode (prodi dominan)": "; ".join(
         f"{k} ({g[PRODI_COL].value_counts().index[0]})"
         for k, g in df[df["Nama MK"] == n].groupby("Kode MK"))}
    for n in nama_multi.index])
simpan_tabel(tabel_nama_multi, "tabel_3_5_nama_sama_kode_berbeda")
catatan["nama_mk_dengan_>1_kode (dipertahankan terpisah)"] = int(len(nama_multi))


# =====================================================================
# III.6 PRODI ACUAN
# =====================================================================
judul("III.6 PENENTUAN PRODI ACUAN")
beda = int((df["Kode Prodi"] != df["Kode Prodi Kelas"]).sum())
print(f"   Record dengan Kode Prodi != Kode Prodi Kelas: {beda} "
      f"({df.loc[df['Kode Prodi'] != df['Kode Prodi Kelas'], 'NIM'].nunique()} mahasiswa)")

prodi_mk = df.groupby("Kode MK")[PRODI_COL].agg(
    prodi_utama=lambda s: s.value_counts().index[0],
    jumlah_prodi=lambda s: s.nunique(),
    porsi_prodi_utama=lambda s: round(s.value_counts(normalize=True).iloc[0], 3))
prodi_mk["multi_prodi"] = prodi_mk["jumlah_prodi"] > 1
pemetaan = pd.DataFrame({
    "Nama MK": baku,
    "Prodi utama": prodi_mk["prodi_utama"],
    "Jumlah prodi kelas": prodi_mk["jumlah_prodi"],
    "Porsi prodi utama": prodi_mk["porsi_prodi_utama"],
    "MK multi-prodi": prodi_mk["multi_prodi"],
    "Pernah bertanda asterisk": df.groupby("Kode MK")["ada_asterisk"].any(),
    "Jumlah mahasiswa": df.groupby("Kode MK")["NIM"].nunique(),
}).reset_index().rename(columns={"index": "Kode MK"})
simpan_tabel(pemetaan, "tabel_3_6_pemetaan_mk_dan_prodi")
n_multi = int(prodi_mk["multi_prodi"].sum())
print(f"   MK yang dikelola >1 prodi kelas: {n_multi} dari {len(prodi_mk)} "
      f"(prodi utama = prodi dengan record terbanyak)")
catatan["record_prodi_beda_prodi_kelas"] = beda
catatan["mk_multi_prodi"] = n_multi
catat("III.6", f"Prodi acuan = {PRODI_COL} (prodi utama per MK; {n_multi} MK multi-prodi ditandai)", df)


# =====================================================================
# III.7 VALIDASI AKHIR
# =====================================================================
judul("III.7 VALIDASI AKHIR")
dup = int(df.duplicated(["NIM", "Kode MK"]).sum())
if dup:
    print(f"   {dup} duplikat (NIM, Kode MK) -> rata-ratakan nilainya")
    df = (df.groupby(["NIM", "Kode MK"], as_index=False)
          .agg({**{c: "first" for c in df.columns if c not in ("NIM", "Kode MK", "Nilai Angka")},
                "Nilai Angka": "mean"}))
huruf_hitung = df["Nilai Angka"].map(huruf_dari_angka)
tak_konsisten = int((huruf_hitung != df["Nilai Huruf"]).sum())
rentang_ok = int(((df["Nilai Angka"] < 0) | (df["Nilai Angka"] > 100)).sum())
print(f"   Duplikat (NIM, Kode MK): {dup} | Nilai Huruf tidak konsisten dengan angka: {tak_konsisten} "
      f"| angka di luar 0-100: {rentang_ok}")
catatan["duplikat_akhir"] = dup
catatan["huruf_angka_tidak_konsisten"] = tak_konsisten
catat("III.7", "Validasi: duplikat dan konsistensi Nilai Huruf vs Nilai Angka", df)


# =====================================================================
# III.8 TRANSFORMASI: MATRIKS MAHASISWA x MK
# =====================================================================
judul("III.8 TRANSFORMASI DATA")
M = df.pivot(index="NIM", columns="Kode MK", values="Nilai Angka")
kepadatan = M.notna().values.mean() * 100
print(f"   Matriks mahasiswa x MK: {M.shape[0]} x {M.shape[1]} | terisi {kepadatan:.2f}% "
      f"(sisanya kosong karena mahasiswa tidak mengambil MK tsb)")
catatan["matriks_mahasiswa"] = M.shape[0]
catatan["matriks_mk"] = M.shape[1]
catatan["matriks_terisi_persen"] = round(kepadatan, 2)
contoh_baris = df[["NIM", "Kode MK", "Nilai Angka"]].head(6)
simpan_tabel(contoh_baris, "tabel_3_8a_contoh_format_long")
simpan_tabel(M.iloc[:6, :6].reset_index(), "tabel_3_8b_contoh_matriks_mahasiswa_mk")

kolom_bersih = ["NIM", "Kode MK", "Nama MK", "Semester", "Kelas", "Nilai Huruf", "Nilai Indeks",
                "Nilai Angka", "Kode Prodi", "Kode Prodi Kelas", "Total SKS", "ada_asterisk"]
df[kolom_bersih].to_csv(DATA_DIR / "data_bersih.csv", index=False, encoding="utf-8-sig")
M.to_csv(DATA_DIR / "matriks_mahasiswa_mk.csv", encoding="utf-8-sig")
pemetaan.to_csv(DATA_DIR / "pemetaan_mk.csv", index=False, encoding="utf-8-sig")
print(f"   [data]   {DATA_DIR / 'data_bersih.csv'}\n   [data]   {DATA_DIR / 'matriks_mahasiswa_mk.csv'}"
      f"\n   [data]   {DATA_DIR / 'pemetaan_mk.csv'}")


# =====================================================================
# TABEL ALUR JUMLAH DATA, CONTOH, GAMBAR
# =====================================================================
judul("RINGKASAN HASIL PREPROCESSING")
tabel_alur = pd.DataFrame(alur)
print(tabel_alur[["Langkah", "Record dikeluarkan", "Record tersisa",
                  "Mahasiswa tersisa", "Kode MK tersisa"]].to_string(index=False))
simpan_tabel(tabel_alur, "tabel_3_0_alur_jumlah_data")
simpan_tabel(pd.DataFrame(contoh, columns=["Jenis perbaikan", "Kolom", "Sebelum", "Sesudah"]),
             "tabel_3_contoh_sebelum_sesudah")

catatan["record_akhir"] = len(df)
catatan["mahasiswa_akhir"] = df["NIM"].nunique()
catatan["kode_mk_akhir"] = df["Kode MK"].nunique()
catatan["prodi_akhir"] = df[PRODI_COL].nunique()
catatan["record_dikeluarkan_total"] = catatan["record_awal"] - len(df)
catatan["persen_dikeluarkan"] = round(catatan["record_dikeluarkan_total"] / catatan["record_awal"] * 100, 2)

# Gambar 1: Nilai Angka sebelum vs sesudah
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].hist(angka_sebelum.dropna(), bins=30, color=BIRU, edgecolor="white")
ax[0].set(title=f"Sebelum (N={angka_sebelum.notna().sum()})", xlabel="Nilai Angka", ylabel="Jumlah record")
ax[1].hist(df["Nilai Angka"], bins=30, color=BIRU, edgecolor="white")
ax[1].set(title=f"Sesudah preprocessing (N={len(df)})", xlabel="Nilai Angka")
for a, serie in zip(ax, (angka_sebelum.dropna(), df["Nilai Angka"])):
    a.axvline(serie.mean(), color=MERAH, ls="--", label=f"Rata-rata {serie.mean():.1f}")
    a.axvline(serie.median(), color="black", ls=":", label=f"Median {serie.median():.1f}")
    a.legend()
fig.suptitle("Distribusi Nilai Angka Sebelum dan Sesudah Preprocessing", fontweight="bold")
simpan_gambar(fig, "gambar_3_1_nilai_sebelum_sesudah")

# Gambar 2: alur jumlah record
tampil = tabel_alur[tabel_alur["Langkah"].isin(["Awal", "III.3", "III.4a", "III.4b", "III.7"])]
fig, ax = plt.subplots(figsize=(8, 3.8))
bars = ax.bar(tampil["Langkah"], tampil["Record tersisa"], color=BIRU)
for b, v in zip(bars, tampil["Record tersisa"]):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}".replace(",", "."), ha="center", va="bottom", fontsize=9)
ax.set(title="Jumlah Record Tersisa pada Tiap Tahap", ylabel="Jumlah record")
ax.set_ylim(0, tampil["Record tersisa"].max() * 1.12)
simpan_gambar(fig, "gambar_3_2_alur_jumlah_record")

# Catatan angka
with open(OUTPUT_DIR / "catatan_angka_preprocessing.txt", "w", encoding="utf-8") as f:
    f.write(f"Sumber: {FILE_INPUT}\nMODE_E: {MODE_E}\nKolom prodi: {PRODI_COL}\n\n")
    for k, v in catatan.items():
        f.write(f"{k}: {v}\n")
print("\nCatatan angka:")
for k, v in catatan.items():
    print(f"   {k}: {v}")
print(f"\n   [file]   {OUTPUT_DIR / 'catatan_angka_preprocessing.txt'}")
print("\nSELESAI. Hasil ada di folder:", OUTPUT_DIR.resolve())
