import os, glob
import numpy as np, cv2, streamlit as st
from skimage.feature import hog, local_binary_pattern

ROOT = "Tobacco3482-jpg"          
H, W = 256, 192

st.set_page_config(page_title="CBIR documents", layout="wide")

# ---------- prétraitement + descripteurs ----------
def prep(g):
    return cv2.resize(g, (W, H), interpolation=cv2.INTER_AREA)

def binarize(g):
    _, b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return b // 255

def f_layout(g, b):  return cv2.resize(g, (12, 16), interpolation=cv2.INTER_AREA).flatten() / 255.
def f_density(g, b): return cv2.resize(b.astype(np.float32), (6, 8), interpolation=cv2.INTER_AREA).flatten()
def f_proj(g, b):
    h = cv2.resize(b.mean(1).reshape(-1, 1).astype(np.float32), (1, 32), interpolation=cv2.INTER_AREA).flatten()
    v = cv2.resize(b.mean(0).reshape(1, -1).astype(np.float32), (32, 1), interpolation=cv2.INTER_AREA).flatten()
    return np.concatenate([h, v])
def f_hog(g, b):
    return hog(g, orientations=9, pixels_per_cell=(32, 32), cells_per_block=(2, 2), feature_vector=True)
def f_lbp(g, b):
    l = local_binary_pattern(g, 8, 1, "uniform")
    h, _ = np.histogram(l, bins=10, range=(0, 10)); return h / h.sum()
def f_edges(g, b):
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    h, _ = np.histogram(ang % 180, bins=12, range=(0, 180), weights=mag)
    return h / (h.sum() + 1e-9)

DESC = {"HOG (forme)": f_hog, "Layout (miniature)": f_layout, "Densité d'encre": f_density,
        "Projections H/V": f_proj, "LBP (texture)": f_lbp, "Contours (orientation)": f_edges}

def describe(g):
    g = prep(g); b = binarize(g)
    return {k: f(g, b) for k, f in DESC.items()}

# ---------- indexation (calculée une fois, puis mise en cache) ----------
@st.cache_data(show_spinner="Indexation du corpus (une seule fois)...")
def build_index():
    paths = sorted(glob.glob(ROOT + "/*/*.jpg"))
    labels = np.array([os.path.basename(os.path.dirname(p)) for p in paths])
    F = {k: [] for k in DESC}
    for p in paths:
        d = describe(cv2.imread(p, cv2.IMREAD_GRAYSCALE))
        for k in DESC: F[k].append(d[k])
    F = {k: np.array(v) for k, v in F.items()}
    stats = {k: (v.mean(0), v.std(0) + 1e-9) for k, v in F.items()}
    return paths, labels, F, stats

def block(x, k, stats):
    m, s = stats[k]
    return ((x - m) / s) / np.sqrt(len(m))     # chaque descripteur pèse autant

def distances(q, X, metric):
    if metric == "Cosinus":
        qn = q / (np.linalg.norm(q) + 1e-9); Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
        return 1 - Xn @ qn
    return np.linalg.norm(X - q, axis=1)

# ---------- interface ----------
st.title("Indexation d'images de documents (CBIR)")
st.caption("Recherche de pages ou formulaires similaires à partir de leurs caractéristiques visuelles. Corpus : Tobacco-3482")

if not os.path.isdir(ROOT):
    st.error(f"Dossier « {ROOT} » introuvable : place-le à côté de app.py."); st.stop()

paths, labels, F, stats = build_index()

with st.sidebar:
    st.header("Paramètres")
    keys = st.multiselect("Descripteurs", list(DESC), default=list(DESC))
    metric = st.radio("Distance", ["Cosinus", "Euclidienne"])
    k = st.slider("Nombre de résultats (k)", 1, 12, 6)
    source = st.radio("Image requête", ["Choisir dans le corpus", "Importer une image"])
    st.write(f"{len(paths)} images indexées")

if not keys:
    st.warning("Choisis au moins un descripteur."); st.stop()

X = np.hstack([block(F[c], c, stats) for c in keys])
q_idx, q_img = None, None

if source == "Choisir dans le corpus":
    cls = st.sidebar.selectbox("Classe", sorted(set(labels)))
    ids = np.where(labels == cls)[0]
    n = st.sidebar.number_input("Numéro de l'image dans la classe", 1, len(ids), 1)
    q_idx = ids[n - 1]
    q_img = cv2.imread(paths[q_idx], cv2.IMREAD_GRAYSCALE)
    d = describe(q_img)
    q = np.hstack([block(d[c], c, stats) for c in keys])
else:
    up = st.sidebar.file_uploader("Image (jpg/png)", type=["jpg", "jpeg", "png"])
    if up is None:
        st.info("Importe une image dans la barre latérale."); st.stop()
    q_img = cv2.imdecode(np.frombuffer(up.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    d = describe(q_img)
    q = np.hstack([block(d[c], c, stats) for c in keys])

dist = distances(q, X, metric)
if q_idx is not None: dist[q_idx] = np.inf          # exclure la requête elle-même
top = np.argsort(dist)[:k]

left, right = st.columns([1, 3])
with left:
    st.subheader("Requête")
    st.image(q_img, width="stretch", caption=labels[q_idx] if q_idx is not None else "image importée")
with right:
    st.subheader("Résultats")
    cols = st.columns(min(k, 4))
    for i, r in enumerate(top):
        with cols[i % len(cols)]:
            ok = q_idx is not None and labels[r] == labels[q_idx]
            tag = ("✅ " if ok else "❌ ") if q_idx is not None else ""
            st.image(prep(cv2.imread(paths[r], cv2.IMREAD_GRAYSCALE)), width="stretch",
                     caption=f"{tag}#{i+1} {labels[r]} (d={dist[r]:.3f})")

if q_idx is not None:
    st.success(f"Précision@{k} pour cette requête : {np.mean(labels[top] == labels[q_idx]):.2f}")
