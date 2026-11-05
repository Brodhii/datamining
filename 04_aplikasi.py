"""
04_aplikasi.py  -  BAB VI: Penerapan Model (aplikasi untuk Pengelola Program Studi)

Aplikasi ini MENJALANKAN ULANG model yang sama dengan 03_model_evaluasi.py
(korelasi Spearman -> koreksi FDR Benjamini-Hochberg -> graf -> Louvain) dan membiarkan
pengguna mengubah ambang secara interaktif. Dengan nilai awal bawaan (n=30, rho=0,30,
q=0,05, SD=1) hasilnya sama dengan laporan: 113 edge, 14 komunitas, modularity 0,868.

Cara menjalankan:
    pip install streamlit pandas numpy scipy networkx matplotlib
    streamlit run 04_aplikasi.py

Masukan : output/data/data_bersih.csv (hasil 02_preprocessing.py), atau unggah lewat sidebar.
"""

import io
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

# =====================================================================
# KONSTANTA
# =====================================================================
DATA_BAWAAN = Path("output/data/data_bersih.csv")
KOLOM_WAJIB = ["NIM", "Kode MK", "Nama MK", "Nilai Angka", "Kode Prodi", "Kode Prodi Kelas"]
N_LANTAI = 20            # korelasi dihitung sekali untuk n >= 20; slider tidak boleh di bawahnya
N_SEED = 20              # Louvain dijalankan 20 seed, dipakai partisi bermodularity tertinggi
RHO_KUAT = 0.50
KAT_DALAM = "Dalam prodi (kedua MK satu prodi)"
KAT_MULTI = "Melibatkan MK multi-prodi"
KAT_LINTAS = "Lintas prodi (kedua MK satu prodi, prodi berbeda)"
WARNA_MULTI = "#e08e0b"

# Nama prodi (opsional): file prodi_nama.csv (kolom Kode Prodi, Nama Prodi) di folder yang sama dengan data_bersih.csv
NAMA_PRODI = {}
_f_prodi = DATA_BAWAAN.parent / "prodi_nama.csv"
if _f_prodi.exists():
    _pn = pd.read_csv(_f_prodi, dtype=str).fillna("")
    NAMA_PRODI = {r["Kode Prodi"]: r["Nama Prodi"].strip() for _, r in _pn.iterrows() if r["Nama Prodi"].strip()}


def label_prodi(kode):
    return f"{NAMA_PRODI[kode]} ({kode})" if kode in NAMA_PRODI else str(kode)


st.set_page_config(page_title="Peta Keterkaitan Nilai Antar Mata Kuliah", layout="wide")

# Parameter lebar berbeda antar versi Streamlit
_V = tuple(int(x) for x in st.__version__.split(".")[:2])
LEBAR = {"width": "stretch"} if _V >= (1, 50) else {"use_container_width": True}


# =====================================================================
# MODEL (sama dengan 03_model_evaluasi.py)
# =====================================================================
def bh_fdr(p):
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


@st.cache_data(show_spinner="Menghitung korelasi Spearman antar pasangan mata kuliah...")
def hitung_pasangan(df, n_lantai):
    """Korelasi Spearman untuk semua pasangan MK dengan n mahasiswa bersama >= n_lantai."""
    M = df.pivot_table(index="NIM", columns="Kode MK", values="Nilai Angka", aggfunc="mean")
    sd = M.std()
    M = M[sd[sd > 0].index]
    kolom = list(M.columns)
    arr = M.values
    ada = (~np.isnan(arr)).astype(int)
    co = ada.T @ ada
    prodi_mhs = (df.groupby("NIM")["Kode Prodi"].agg(lambda s: s.value_counts().index[0])
                 .reindex(M.index).values)
    sd_all = M.std().values

    meta = pd.DataFrame({
        "Nama MK": df.groupby("Kode MK")["Nama MK"].agg(lambda s: s.value_counts().index[0]),
        "Prodi utama": df.groupby("Kode MK")["Kode Prodi Kelas"].agg(lambda s: s.value_counts().index[0]),
        "Jumlah prodi": df.groupby("Kode MK")["Kode Prodi Kelas"].nunique(),
        "Jumlah mahasiswa": df.groupby("Kode MK")["NIM"].nunique(),
    })
    meta["Multi-prodi"] = meta["Jumlah prodi"] > 1

    baris = []
    for i, j in np.argwhere(np.triu(co >= n_lantai, 1)):
        m = ~np.isnan(arr[:, i]) & ~np.isnan(arr[:, j])
        x, y = arr[m, i], arr[m, j]
        if np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        r, p = stats.spearmanr(x, y)
        if np.isnan(r):
            continue
        porsi = np.unique(prodi_mhs[m], return_counts=True)[1].max() / m.sum()
        baris.append((kolom[i], kolom[j], int(m.sum()), r, p, sd_all[i], sd_all[j], porsi))
    P = pd.DataFrame(baris, columns=["a", "b", "n", "rho", "p", "sd_a", "sd_b", "porsi_prodi_mhs"])
    P["nama_a"] = P["a"].map(meta["Nama MK"])
    P["nama_b"] = P["b"].map(meta["Nama MK"])
    P["prodi_a"] = P["a"].map(meta["Prodi utama"])
    P["prodi_b"] = P["b"].map(meta["Prodi utama"])
    P["multi"] = P["a"].map(meta["Multi-prodi"]) | P["b"].map(meta["Multi-prodi"])
    P["kategori"] = np.select([P["multi"], P["prodi_a"] == P["prodi_b"]], [KAT_MULTI, KAT_DALAM],
                              default=KAT_LINTAS)
    return P, meta, M.shape


@st.cache_data(show_spinner="Menjalankan koreksi FDR dan Louvain...")
def jalankan_model(P, n_min, rho_min, sd_min, q_fdr):
    T = P[(P["n"] >= n_min) & (P["sd_a"] >= sd_min) & (P["sd_b"] >= sd_min)].copy()
    if T.empty:
        return T, T, T, {}, np.nan, 0, 0
    T["q"] = bh_fdr(T["p"].values)
    T["sig"] = T["q"] < q_fdr
    E = T[T["sig"] & (T["rho"] >= rho_min)].copy()
    Eneg = T[T["sig"] & (T["rho"] <= -rho_min)].copy()
    if E.empty:
        return T, E, Eneg, {}, np.nan, 0, 0
    G = nx.Graph()
    G.add_weighted_edges_from(zip(E["a"], E["b"], E["rho"]))
    terbaik = None
    for s in range(N_SEED):
        kom = nx.community.louvain_communities(G, weight="weight", seed=s)
        q = nx.community.modularity(G, kom, weight="weight")
        if terbaik is None or q > terbaik[1] + 1e-12:
            terbaik = (kom, q)
    kom, Q = terbaik
    urut = sorted(kom, key=lambda c: (-len(c), min(c)))
    part = {n: i + 1 for i, c in enumerate(urut) for n in c}
    return T, E, Eneg, part, Q, nx.number_connected_components(G), len(kom)


def ringkasan_komunitas(E, part, meta):
    G = nx.Graph()
    G.add_weighted_edges_from(zip(E["a"], E["b"], E["rho"]))
    baris = []
    for c in sorted(set(part.values())):
        nodes = [n for n in part if part[n] == c]
        sub = G.subgraph(nodes)
        tunggal = [n for n in nodes if not meta.loc[n, "Multi-prodi"]] or nodes
        pr = pd.Series([meta.loc[n, "Prodi utama"] for n in tunggal])
        baris.append({
            "Komunitas": c, "Jumlah MK": len(nodes),
            "MK multi-prodi": int(sum(meta.loc[n, "Multi-prodi"] for n in nodes)),
            "Jumlah edge": sub.number_of_edges(),
            "Rata-rata rho": round(np.mean([d["weight"] for _, _, d in sub.edges(data=True)]), 3)
            if sub.number_of_edges() else np.nan,
            "Prodi dominan": label_prodi(pr.value_counts().index[0]),
            "Porsi prodi dominan": round(pr.value_counts(normalize=True).iloc[0], 2),
        })
    return pd.DataFrame(baris)


def gambar_jaringan(E, part, meta, tampil, label_nama):
    G = nx.Graph()
    G.add_weighted_edges_from(zip(E["a"], E["b"], E["rho"]))
    nodes_tampil = [n for n in G.nodes() if part[n] in tampil]
    H = G.subgraph(nodes_tampil)
    kelompok = {}
    for n in nodes_tampil:
        kelompok.setdefault(part[n], []).append(n)
    ncol = max(1, int(np.ceil(np.sqrt(len(kelompok)))))
    pos = {}
    for k, (c, nodes) in enumerate(sorted(kelompok.items())):
        cx, cy = (k % ncol) * 4.2, -(k // ncol) * 4.0
        if len(nodes) == 1:
            loc = {nodes[0]: (0, 0)}
        elif len(nodes) == 2:
            loc = {nodes[0]: (-0.5, 0), nodes[1]: (0.5, 0)}
        else:
            loc = nx.spring_layout(H.subgraph(nodes), seed=42, scale=min(1.7, 0.55 + 0.11 * len(nodes)),
                                   k=2.2 / np.sqrt(len(nodes)), iterations=200)
        for n, (x, y) in loc.items():
            pos[n] = (cx + x, cy + y)
    palet = plt.get_cmap("tab20")
    multi = lambda n: bool(meta.loc[n, "Multi-prodi"])
    e_multi = [(u, v) for u, v in H.edges() if multi(u) or multi(v)]
    e_biasa = [(u, v) for u, v in H.edges() if not (multi(u) or multi(v))]
    fig, ax = plt.subplots(figsize=(14, 9.5))
    nx.draw_networkx_edges(H, pos, edgelist=e_biasa, ax=ax, edge_color="#8c8c8c", width=1.1)
    nx.draw_networkx_edges(H, pos, edgelist=e_multi, ax=ax, edge_color=WARNA_MULTI, width=1.8, style="dashed")
    nx.draw_networkx_nodes(H, pos, ax=ax, node_color=[palet((part[n] - 1) % 20) for n in H.nodes()],
                           edgecolors="white", node_size=[120 + 60 * H.degree(n) for n in H.nodes()])
    etiket = {n: (textwrap.fill(str(meta.loc[n, "Nama MK"]), 16) if label_nama else n) for n in H.nodes()}
    nx.draw_networkx_labels(H, pos, labels=etiket, ax=ax, font_size=6 if label_nama else 6.5)
    for c, nodes in kelompok.items():
        xs, ys = [pos[n][0] for n in nodes], [pos[n][1] for n in nodes]
        ax.text(np.mean(xs), max(ys) + 0.3, f"Komunitas {c} ({len(nodes)} MK)", ha="center",
                fontsize=9, fontweight="bold")
    ax.legend(handles=[plt.Line2D([0], [0], color="#8c8c8c", lw=1.5, label="Edge antar MK satu prodi"),
                       plt.Line2D([0], [0], color=WARNA_MULTI, lw=2, ls="--",
                                  label="Edge yang melibatkan MK multi-prodi (mis. MK umum)")],
              loc="lower right")
    ax.axis("off")
    fig.tight_layout()
    return fig


def bungkus(teks, lebar=16):
    return textwrap.fill(str(teks), lebar)


def tata_letak_inci(Hc, w, h):
    """Tata letak satu komunitas (satuan inci); titik digeser sampai kotak label (w x h) tidak tumpang tindih."""
    nodes = list(Hc.nodes())
    n = len(nodes)
    if n == 1:
        return {nodes[0]: np.zeros(2)}
    if n == 2:
        return {nodes[0]: np.array([-w * 0.75, 0.0]), nodes[1]: np.array([w * 0.75, 0.0])}
    try:
        p0 = nx.kamada_kawai_layout(Hc)
    except Exception:
        p0 = nx.spring_layout(Hc, seed=42)
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


def gambar_komunitas_detail(E, part, meta, c, rho_min):
    """Peta satu komunitas dengan NAMA MK yang terbaca (kode MK kecil di bawah nama)."""
    nodes = [n for n in part if part[n] == c]
    G = nx.Graph()
    G.add_weighted_edges_from(zip(E["a"], E["b"], E["rho"]))
    H = G.subgraph(nodes)
    multi = lambda n: bool(meta.loc[n, "Multi-prodi"])
    nama = {n: bungkus(meta.loc[n, "Nama MK"]) for n in nodes}
    maks_char = max(len(l) for t in nama.values() for l in t.split("\n"))
    maks_baris = max(t.count("\n") + 1 for t in nama.values())
    w = max(1.5, maks_char * 0.085 + 0.4)
    h = (maks_baris + 1) * 0.17 + 0.55
    pos = tata_letak_inci(H, w, h)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    x0, x1 = min(xs) - w / 2 - 0.3, max(xs) + w / 2 + 0.3
    y0, y1 = min(ys) - h - 0.1, max(ys) + 0.5
    fig, ax = plt.subplots(figsize=(max(6.0, x1 - x0), max(4.0, y1 - y0 + 0.7)))
    warna = plt.get_cmap("tab20")((c - 1) % 20)
    for u, v, d in H.edges(data=True):
        ada_multi = multi(u) or multi(v)
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], zorder=1,
                color=WARNA_MULTI if ada_multi else "#8c8c8c", ls="--" if ada_multi else "-",
                lw=1.0 + (d["weight"] - rho_min) * 9, alpha=0.75)
    kotak = dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.8)
    for n in nodes:
        x, y = pos[n]
        ax.scatter([x], [y], s=150 + 70 * H.degree(n), color=warna, edgecolor="white", linewidth=1.5,
                   zorder=3, marker="D" if multi(n) else "o")
        ax.text(x, y - 0.17, nama[n], ha="center", va="top", fontsize=9, zorder=4, fontweight="bold", bbox=kotak)
        ax.text(x, y - 0.17 - (nama[n].count("\n") + 1) * 0.165 - 0.02, n, ha="center", va="top",
                fontsize=6.5, color="#666666", zorder=4, bbox=kotak)
    pr = pd.Series([meta.loc[n, "Prodi utama"] for n in nodes if not multi(n)] or
                   [meta.loc[n, "Prodi utama"] for n in nodes])
    ax.set_title(f"Komunitas {c}: {len(nodes)} mata kuliah | {H.number_of_edges()} keterkaitan | "
                 f"prodi dominan {label_prodi(pr.value_counts().index[0])}", fontsize=11, fontweight="bold")
    ax.legend(handles=[plt.Line2D([0], [0], color="#8c8c8c", lw=3, label="Keterkaitan (tebal garis = kekuatan rho)"),
                       plt.Line2D([0], [0], color=WARNA_MULTI, lw=2, ls="--", label="Melibatkan MK multi-prodi"),
                       plt.Line2D([0], [0], marker="D", color="w", markerfacecolor=warna, markersize=8,
                                  label="MK multi-prodi (mis. MK umum)")],
              loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7, frameon=False)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.axis("off")
    return fig


def csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8-sig")


# =====================================================================
# SIDEBAR: DATA DAN PARAMETER
# =====================================================================
st.sidebar.title("Pengaturan")
st.sidebar.subheader("1. Data")
unggah = st.sidebar.file_uploader("Unggah data_bersih.csv (opsional)", type="csv")
dtype = {"NIM": str, "Kode MK": str, "Kode Prodi": str, "Kode Prodi Kelas": str}
if unggah is not None:
    df = pd.read_csv(unggah, dtype=dtype)
elif DATA_BAWAAN.exists():
    df = pd.read_csv(DATA_BAWAAN, dtype=dtype)
else:
    st.error(f"File {DATA_BAWAAN} tidak ditemukan. Jalankan 02_preprocessing.py lebih dulu, "
             "atau unggah data_bersih.csv lewat sidebar.")
    st.stop()
kurang = [k for k in KOLOM_WAJIB if k not in df.columns]
if kurang:
    st.error(f"Kolom berikut tidak ada di data: {kurang}")
    st.stop()

st.sidebar.subheader("2. Parameter analisis")
n_min = st.sidebar.slider("Jumlah mahasiswa bersama minimum (n)", N_LANTAI, 60, 30, 5,
                          help="Pasangan MK dengan mahasiswa bersama lebih sedikit dari ini tidak diuji.")
rho_min = st.sidebar.slider("Korelasi minimum (rho)", 0.10, 0.80, 0.30, 0.05,
                            help="Batas bawah korelasi Spearman agar sebuah pasangan menjadi edge.")
q_fdr = st.sidebar.select_slider("Batas FDR (q)", options=[0.01, 0.05, 0.10], value=0.05,
                                 help="Koreksi multiple comparisons Benjamini-Hochberg.")
sd_min = st.sidebar.slider("SD nilai minimum per MK", 0.0, 3.0, 1.0, 0.5,
                           help="MK dengan nilai hampir seragam dikeluarkan.")
st.sidebar.caption("Nilai awal (n=30, rho=0,30, q=0,05, SD=1) sama dengan parameter di laporan.")
label_nama = st.sidebar.radio("Label pada peta", ["Kode MK", "Nama MK"], horizontal=True) == "Nama MK"

# =====================================================================
# JALANKAN MODEL
# =====================================================================
P, meta, bentuk_matriks = hitung_pasangan(df, N_LANTAI)
T, E, Eneg, part, Q, n_komponen, n_kom = jalankan_model(P, n_min, rho_min, sd_min, q_fdr)

st.title("Peta Keterkaitan Capaian Nilai Antar Mata Kuliah")
st.caption("Korelasi Spearman Nilai Angka, koreksi FDR, dan deteksi komunitas Louvain. "
           "Hasil bersifat deskriptif sebagai informasi pendukung peninjauan struktur mata kuliah "
           "dan tidak menunjukkan hubungan sebab-akibat.")

if T.empty or E.empty:
    st.warning("Tidak ada edge yang lolos dengan parameter ini. Turunkan korelasi minimum, "
               "kurangi n minimum, atau longgarkan SD minimum.")
    st.stop()

k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Pasangan MK diuji", len(T))
k2.metric("Signifikan (FDR)", int(T["sig"].sum()))
k3.metric("Edge jaringan utama", len(E))
k4.metric("MK dalam jaringan", len(part))
k5.metric("Komunitas", n_kom)
k6.metric("Modularity", f"{Q:.3f}")

tab = st.tabs(["Ringkasan", "Peta jaringan", "Komunitas", "Pasangan MK", "Cari mata kuliah",
               "Dalam vs lintas prodi", "Korelasi negatif", "Catatan"])

# ---------------------------------------------------------------- Ringkasan
with tab[0]:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Data yang dianalisis")
        st.dataframe(pd.DataFrame({
            "Keterangan": ["Record", "Mahasiswa", "Mata kuliah", "Prodi (kelas)",
                           "MK dengan variasi nilai (SD > 0)", "Kemungkinan pasangan MK"],
            "Jumlah": [len(df), df["NIM"].nunique(), df["Kode MK"].nunique(),
                       df["Kode Prodi Kelas"].nunique(), bentuk_matriks[1],
                       bentuk_matriks[1] * (bentuk_matriks[1] - 1) // 2]}),
            hide_index=True, **LEBAR)
    with c2:
        st.subheader("Hasil dengan parameter saat ini")
        st.dataframe(pd.DataFrame({
            "Keterangan": ["Pasangan diuji (n >= batas)", "Signifikan setelah FDR", "  - korelasi positif",
                           "  - korelasi negatif", "Edge jaringan utama (positif, rho >= batas)",
                           "Komponen terhubung", "Komunitas Louvain", "Modularity"],
            "Nilai": [len(T), int(T["sig"].sum()), int((T["sig"] & (T["rho"] > 0)).sum()),
                      int((T["sig"] & (T["rho"] < 0)).sum()), len(E), n_komponen, n_kom, round(Q, 3)]}),
            hide_index=True, **LEBAR)
    st.info(f"Dari {len(T)} pasangan yang diuji, {len(E)} pasangan menjadi edge. "
            f"Jaringan terdiri dari {len(part)} mata kuliah dalam {n_kom} komunitas. "
            f"Ada {n_komponen} komponen terhubung, sehingga sebagian besar komunitas sama dengan satu "
            "komponen. Artinya peta ini menunjukkan gugus mata kuliah yang saling terhubung, bukan "
            "hierarki kurikulum.")
    if n_komponen >= 0.8 * n_kom:
        st.caption("Komponen terhubung hampir sama banyak dengan komunitas: Louvain hanya sedikit "
                   "memecah komponen yang sudah ada.")

# ---------------------------------------------------------------- Peta
with tab[1]:
    ring = ringkasan_komunitas(E, part, meta)
    pilihan = st.multiselect("Tampilkan komunitas", sorted(set(part.values())),
                             default=sorted(set(part.values())))
    if len(pilihan) == 1:
        st.caption("Tampilan detail satu komunitas: nama mata kuliah ditampilkan jelas (kode MK di bawah nama). "
                   "Cocok untuk presentasi.")
        fig = gambar_komunitas_detail(E, part, meta, pilihan[0], rho_min)
        st.pyplot(fig, **LEBAR)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
        st.download_button("Unduh peta komunitas (PNG)", buf.getvalue(), f"komunitas_{pilihan[0]:02d}.png", "image/png")
        plt.close(fig)
    elif pilihan:
        st.caption("Peta ringkas semua komunitas. Agar nama mata kuliah terbaca jelas, pilih **satu komunitas** "
                   "pada kotak di atas.")
        fig = gambar_jaringan(E, part, meta, set(pilihan), label_nama)
        st.pyplot(fig, **LEBAR)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        st.download_button("Unduh peta (PNG)", buf.getvalue(), "peta_jaringan_mk.png", "image/png")
        plt.close(fig)
    else:
        st.info("Pilih minimal satu komunitas.")

# ---------------------------------------------------------------- Komunitas
with tab[2]:
    st.subheader("Ringkasan komunitas")
    st.dataframe(ring, hide_index=True, **LEBAR)
    st.download_button("Unduh ringkasan (CSV)", csv_bytes(ring), "ringkasan_komunitas.csv", "text/csv")
    pilih = st.selectbox("Lihat anggota komunitas", sorted(set(part.values())))
    G_tmp = nx.Graph()
    G_tmp.add_weighted_edges_from(zip(E["a"], E["b"], E["rho"]))
    anggota = pd.DataFrame([{
        "Kode MK": n, "Nama MK": meta.loc[n, "Nama MK"], "Prodi utama": label_prodi(meta.loc[n, "Prodi utama"]),
        "MK multi-prodi": bool(meta.loc[n, "Multi-prodi"]), "Jumlah mahasiswa": int(meta.loc[n, "Jumlah mahasiswa"]),
        "Jumlah keterkaitan": G_tmp.degree(n),
    } for n in part if part[n] == pilih]).sort_values("Jumlah keterkaitan", ascending=False)
    st.dataframe(anggota, hide_index=True, **LEBAR)
    semua = pd.DataFrame([{"Komunitas": part[n], "Kode MK": n, "Nama MK": meta.loc[n, "Nama MK"],
                           "Prodi utama": meta.loc[n, "Prodi utama"]} for n in part]).sort_values(["Komunitas", "Kode MK"])
    st.download_button("Unduh semua anggota komunitas (CSV)", csv_bytes(semua), "anggota_komunitas.csv", "text/csv")

# ---------------------------------------------------------------- Pasangan MK
with tab[3]:
    st.subheader("Pasangan mata kuliah yang berkorelasi (jaringan utama)")
    f1, f2, f3 = st.columns(3)
    kat_pilih = f1.multiselect("Kategori", [KAT_DALAM, KAT_MULTI, KAT_LINTAS], default=[KAT_DALAM, KAT_MULTI, KAT_LINTAS])
    cari = f2.text_input("Cari (kode atau nama MK)")
    hanya_kuat = f3.checkbox(f"Hanya korelasi kuat (rho >= {RHO_KUAT})")
    tabel = E.copy()
    tabel["Kuat"] = tabel["rho"] >= RHO_KUAT
    tabel["Komunitas A"] = tabel["a"].map(part)
    tabel["Komunitas B"] = tabel["b"].map(part)
    tabel = tabel[tabel["kategori"].isin(kat_pilih)]
    if hanya_kuat:
        tabel = tabel[tabel["Kuat"]]
    if cari:
        kunci = cari.lower()
        tabel = tabel[tabel[["a", "b", "nama_a", "nama_b"]].apply(
            lambda s: s.str.lower().str.contains(kunci, regex=False)).any(axis=1)]
    tampil = tabel.sort_values("rho", ascending=False)[
        ["a", "nama_a", "b", "nama_b", "n", "rho", "q", "kategori", "Kuat", "Komunitas A", "Komunitas B"]
    ].rename(columns={"a": "Kode MK A", "nama_a": "Nama MK A", "b": "Kode MK B", "nama_b": "Nama MK B",
                      "rho": "rho", "kategori": "Kategori"})
    tampil["rho"] = tampil["rho"].round(3)
    tampil["q"] = tampil["q"].map(lambda v: f"{v:.2e}")
    st.caption(f"{len(tampil)} pasangan ditampilkan")
    st.dataframe(tampil, hide_index=True, **LEBAR)
    st.download_button("Unduh tabel (CSV)", csv_bytes(tampil), "pasangan_mk.csv", "text/csv")

# ---------------------------------------------------------------- Cari MK
with tab[4]:
    st.subheader("Cari keterkaitan sebuah mata kuliah")
    semua_mk = sorted(meta.index)
    opsi = {f"{k} - {meta.loc[k, 'Nama MK']}": k for k in semua_mk}
    pilih_mk = opsi[st.selectbox("Pilih mata kuliah", list(opsi.keys()))]
    st.write(f"**{meta.loc[pilih_mk, 'Nama MK']}** ({pilih_mk}) | prodi utama {meta.loc[pilih_mk, 'Prodi utama']} | "
             f"{int(meta.loc[pilih_mk, 'Jumlah mahasiswa'])} mahasiswa"
             + (" | MK multi-prodi" if meta.loc[pilih_mk, "Multi-prodi"] else ""))
    if pilih_mk in part:
        rekan = [n for n in part if part[n] == part[pilih_mk] and n != pilih_mk]
        st.success(f"Termasuk **komunitas {part[pilih_mk]}** bersama {len(rekan)} MK lain.")
        if rekan:
            st.write("Anggota lain: " + "; ".join(f"{n} ({meta.loc[n, 'Nama MK']})" for n in rekan))
    else:
        st.info("MK ini tidak masuk jaringan utama pada parameter saat ini (tidak punya pasangan "
                "yang lolos n, korelasi, dan FDR).")
    sig = T[T["sig"] & ((T["a"] == pilih_mk) | (T["b"] == pilih_mk))].copy()
    if sig.empty:
        st.write("Tidak ada pasangan signifikan untuk MK ini.")
    else:
        sig["MK lain"] = np.where(sig["a"] == pilih_mk, sig["b"], sig["a"])
        sig["Nama MK lain"] = np.where(sig["a"] == pilih_mk, sig["nama_b"], sig["nama_a"])
        sig["Hubungan"] = np.where(sig["rho"] > 0, "Positif", "Negatif")
        sig["Masuk jaringan utama"] = (sig["rho"] >= rho_min)
        out = sig.sort_values("rho", ascending=False)[["MK lain", "Nama MK lain", "n", "rho", "Hubungan",
                                                       "Masuk jaringan utama", "kategori"]]
        out["rho"] = out["rho"].round(3)
        st.dataframe(out.rename(columns={"kategori": "Kategori"}), hide_index=True, **LEBAR)

# ---------------------------------------------------------------- Dalam vs lintas
with tab[5]:
    st.subheader("Dalam prodi vs lintas prodi")
    is_edge = T["sig"] & (T["rho"] >= rho_min)
    is_kuat = T["sig"] & (T["rho"] >= RHO_KUAT)
    baris = []
    for kat in (KAT_DALAM, KAT_MULTI, KAT_LINTAS):
        m = T["kategori"] == kat
        baris.append({
            "Kategori": kat, "Pasangan diuji": int(m.sum()), "Edge": int((m & is_edge).sum()),
            "Persen edge": round((m & is_edge).sum() / m.sum() * 100, 2) if m.sum() else np.nan,
            "Edge kuat": int((m & is_kuat).sum()),
            "Rata-rata rho (edge)": round(T.loc[m & is_edge, "rho"].mean(), 3) if (m & is_edge).any() else np.nan,
            "Rata-rata rho (edge kuat)": round(T.loc[m & is_kuat, "rho"].mean(), 3) if (m & is_kuat).any() else np.nan,
        })
    tab_kat = pd.DataFrame(baris)
    st.dataframe(tab_kat.style.format({"Persen edge": "{:.2f}", "Rata-rata rho (edge)": "{:.3f}",
                                    "Rata-rata rho (edge kuat)": "{:.3f}"}, na_rep="-"),
                 hide_index=True, **LEBAR)
    pur = T["porsi_prodi_mhs"]
    st.write(f"Rata-rata **{pur.mean() * 100:.1f}%** mahasiswa pada tiap pasangan berasal dari satu prodi; "
             f"hanya {int((pur < 0.8).sum())} dari {len(T)} pasangan yang porsi prodi dominannya di bawah 80%.")
    if (T["kategori"] == KAT_LINTAS).sum() == 0:
        st.warning("Tidak ada pasangan lintas prodi sejati (kedua MK khusus satu prodi) yang memenuhi n minimum: "
                   "mahasiswa hampir tidak mengambil MK khusus dari dua prodi sekaligus. Keterkaitan lintas prodi "
                   "hanya terjadi melalui MK multi-prodi (misalnya MK umum universitas dan Skripsi).")

# ---------------------------------------------------------------- Negatif
with tab[6]:
    st.subheader("Korelasi negatif (temuan tambahan)")
    neg = T[T["sig"] & (T["rho"] < 0)].sort_values("rho")[["a", "nama_a", "b", "nama_b", "n", "rho", "q", "kategori"]]
    neg = neg.rename(columns={"a": "Kode MK A", "nama_a": "Nama MK A", "b": "Kode MK B", "nama_b": "Nama MK B",
                              "kategori": "Kategori"})
    neg["rho"] = neg["rho"].round(3)
    neg["q"] = neg["q"].map(lambda v: f"{v:.2e}")
    st.caption("Tidak masuk jaringan utama. Korelasi negatif berarti mahasiswa yang nilainya tinggi di satu MK "
               "cenderung rendah di MK lain, dan bukan berarti satu MK menyebabkan yang lain.")
    st.dataframe(neg, hide_index=True, **LEBAR)

# ---------------------------------------------------------------- Catatan
with tab[7]:
    st.subheader("Cara membaca dan keterbatasan")
    st.markdown(
        """
- **Satuan analisis:** mata kuliah (node). Dua MK terhubung (edge) bila nilai mahasiswa yang mengambil keduanya
  berkorelasi positif (Spearman) cukup kuat, jumlah mahasiswa bersamanya cukup, dan korelasinya tetap signifikan
  setelah koreksi FDR.
- **Komunitas** adalah kelompok MK yang saling terhubung rapat menurut algoritma Louvain.
- **Cakupan terbatas:** hanya MK yang punya cukup mahasiswa bersama yang masuk jaringan. MK lain bukan berarti tidak
  berkaitan, hanya datanya belum cukup.
- **Satu semester:** hasil menggambarkan satu periode dan belum tentu berlaku pada semester lain.
- **Sebab-akibat tidak dapat disimpulkan.** Korelasi dapat muncul karena mahasiswa, kelas, atau dosen yang sama.
- **Komunitas banyak mengikuti prodi** karena mahasiswa jarang mengambil MK khusus dari prodi lain.
- Gunakan hasil ini sebagai **bahan pertimbangan tambahan**, bukan dasar tunggal perubahan kurikulum.
        """)
