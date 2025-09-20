import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.sparse import hstack, csr_matrix
import plotly.express as px
import plotly.graph_objects as go
import openai  # optional: only used if OPENAI_API_KEY is set
from typing import List

# Konfiguration
st.set_page_config(page_title="Filmrekommendationer med Generativ AI", layout="wide")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", None)
if OPENAI_KEY:
    openai.api_key = OPENAI_KEY

# Hjälpfunktioner - caching
@st.cache_data(show_spinner=False)
def load_and_prepare_data(csv_path: str = "Netflix Dataset.csv"):
    df = pd.read_csv(csv_path)
    df = df.rename(columns=lambda s: s.strip())
    movies = df[df['Duration'].astype(str).str.contains("min", na=False)].copy()
    movies['Minutes'] = movies['Duration'].str.extract(r'(\d+)').astype(float, errors='ignore')
    movies['Minutes'] = movies['Minutes'].fillna(movies['Minutes'].median())
    movies['Description'] = movies['Description'].fillna("")

    if 'Release_Year' not in movies.columns:
        if 'Release_Date' in movies.columns:
            movies['Release_Year'] = pd.to_datetime(movies['Release_Date'], errors='coerce').dt.year.fillna(0).astype(int)
        elif 'Year' in movies.columns:
            movies['Release_Year'] = pd.to_numeric(movies['Year'], errors='coerce').fillna(0).astype(int)
        else:
            movies['Release_Year'] = 0

    if 'Type' in movies.columns:
        movies['Genres_list'] = movies['Type'].astype(str).apply(lambda x: [g.strip() for g in x.split(",")] if x else [])
    elif 'Genres' in movies.columns:
        movies['Genres_list'] = movies['Genres'].astype(str).apply(lambda x: [g.strip() for g in x.split(",")] if x else [])
    else:
        movies['Genres_list'] = [[] for _ in range(len(movies))]

    cat_cols = [c for c in ['Category', 'Country', 'Rating'] if c in movies.columns]
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    if cat_cols:
        cat_matrix = encoder.fit_transform(movies[cat_cols].fillna("Unknown"))
    else:
        cat_matrix = csr_matrix((len(movies), 0))

    mlb = MultiLabelBinarizer()
    genres_matrix = mlb.fit_transform(movies['Genres_list'])

    scaler = StandardScaler()
    num_cols = []
    if 'Minutes' in movies.columns:
        num_cols.append('Minutes')
    if 'Release_Year' in movies.columns:
        num_cols.append('Release_Year')
    if num_cols:
        num_matrix = scaler.fit_transform(movies[num_cols].fillna(0))
        num_matrix = csr_matrix(num_matrix)
    else:
        num_matrix = csr_matrix((len(movies), 0))

    tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies['Description'].astype(str))

    X = hstack([tfidf_matrix, cat_matrix, num_matrix, csr_matrix(genres_matrix)])
    X = csr_matrix(X)

    meta = {
        'encoder': encoder,
        'mlb': mlb,
        'tfidf': tfidf,
        'num_cols': num_cols
    }
    movies = movies.reset_index(drop=True)
    return movies, X, meta

@st.cache_resource
def compute_similarity_matrix(_X):
    return cosine_similarity(_X, _X)

@st.cache_resource
def compute_pca(_X, n_components=2, max_samples=2000):
    if _X.shape[0] > max_samples:
        sample = _X[:max_samples].toarray()
    else:
        sample = _X.toarray()
    pca = PCA(n_components=n_components, random_state=42)
    coords = pca.fit_transform(sample)
    return coords, pca

@st.cache_data(show_spinner=False)
def compute_kmeans(coords, n_clusters=8):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)
    return labels, kmeans

# Textgenerering

def generate_reco_text_openai(base_title: str, rec_title: str, context_snippet: str = "") -> str:
    if not OPENAI_KEY:
        raise RuntimeError("OpenAI API key saknas")
    prompt = (
        f"Skriv en kort och lockande rekommendationstext på svenska för filmrekommendation.\n"
        f"Basfilm: {base_title}\n"
        f"Rekommenderad film: {rec_title}\n"
        f"Extra kontext: {context_snippet}\n"
        f"Exempel: 'Om du gillar Stranger Things, så gillar du Dark — båda är...'\n"
        f"Ge 1 mening, max 28 ord."
    )
    resp = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=60,
        temperature=0.8,
        n=1
    )
    return resp.choices[0].text.strip()

def generate_reco_template(base_title: str, rec_title: str, reason: str = "") -> str:
    if reason:
        return f"Om du gillade {base_title}, kan {rec_title} passa — {reason}."
    return f"Om du gillade {base_title}, kanske {rec_title} passar — båda har stark stämning och engagerande berättelser."

@st.cache_data(show_spinner=False)
def generate_reco_texts_batch(base_title: str, recs: List[str], movies_df: pd.DataFrame, use_openai: bool = False):
    results = []
    title_to_desc = movies_df.set_index('Title')['Description'].to_dict()
    for rec in recs:
        snippet = title_to_desc.get(rec, "")[:200]
        if use_openai and OPENAI_KEY:
            try:
                txt = generate_reco_text_openai(base_title, rec, snippet)
            except Exception:
                txt = generate_reco_template(base_title, rec, reason=snippet[:80])
        else:
            txt = generate_reco_template(base_title, rec, reason=snippet[:80])
        results.append(txt)
    return results

# Rekommendationsfunktion
def get_recommendations(title: str, movies: pd.DataFrame, sim_matrix: np.ndarray, top_n: int = 5):
    title_to_index = {t: i for i, t in enumerate(movies['Title'])}
    if title not in title_to_index:
        return pd.DataFrame()
    idx = title_to_index[title]
    sims = sim_matrix[idx]
    order = np.argsort(-sims)
    order = [i for i in order if i != idx]
    top_idx = order[:top_n]
    return movies.iloc[top_idx].reset_index(drop=True)

# SIDOMENY
st.sidebar.header("Inställningar")
csv_path = st.sidebar.text_input("CSV-fil (sökväg)", "Netflix Dataset.csv")
use_openai = st.sidebar.checkbox("Använd OpenAI för genererad text (om API-nyckel finns)", value=False)
top_n = st.sidebar.slider("Antal rekommendationer", 1, 12, 5)
n_clusters = st.sidebar.slider("Antal kluster (KMeans)", 2, 20, 8)
pca_sample_limit = st.sidebar.slider("Visa max N poster i plotten (PCA) för prestanda", 200, 5000, 2000, step=200)
show_pca_cluster = st.sidebar.checkbox("Visa PCA-visualisering & Kluster", value=False)

# Ladda data
with st.spinner("Laddar data och beräknar features..."):
    movies, X, meta = load_and_prepare_data(csv_path)
    sim_matrix = compute_similarity_matrix(X)

# PCA och KMeans förbereds (oavsett om vi visar eller ej)
with st.spinner("Beräknar PCA & kluster..."):
    n_total = X.shape[0]
    if n_total > pca_sample_limit:
        X_small = X[:pca_sample_limit]
        coords_small, pca = compute_pca(X_small, n_components=2)
        try:
            X_dense = X.toarray()
            coords = pca.transform(X_dense)
        except Exception:
            coords = np.zeros((n_total, 2))
            coords[:pca_sample_limit] = coords_small
    else:
        coords, pca = compute_pca(X, n_components=2)
    labels, kmeans = compute_kmeans(coords, n_clusters=n_clusters)
    movies['cluster'] = labels

# HUVUD-UI
st.title("Filmrekommendationer med AI-autocomplete & Klusterutforskning")

selected_title = st.selectbox("Välj film", options=movies['Title'].sort_values().unique())
if not selected_title:
    st.stop()

col1, _ = st.columns([2, 3])
with col1:
    st.subheader(selected_title)
    info_row = movies[movies['Title'] == selected_title].iloc[0]
    st.markdown(f"**År:** {info_row.get('Release_Year', '—')}")
    st.markdown(f"**Längd (min):** {info_row.get('Minutes', '—')}")
    if 'Category' in movies.columns:
        st.markdown(f"**Kategori:** {info_row.get('Category', '—')}")
    if 'Country' in movies.columns:
        st.markdown(f"**Land:** {info_row.get('Country', '—')}")
    st.markdown("**Beskrivning:**")
    st.write(info_row.get('Description', '—'))

    st.markdown("---")
    st.markdown("**Rekommendationer**")
    recs_df = get_recommendations(selected_title, movies, sim_matrix, top_n=top_n)
    st.write(f"Visar {len(recs_df)} rekommendationer (baserat på cosine similarity).")

    with st.spinner("Genererar rekommendationstexter..."):
        rec_titles = recs_df['Title'].tolist()
        reco_texts = generate_reco_texts_batch(selected_title, rec_titles, movies, use_openai=use_openai)
    for title, txt in zip(rec_titles, reco_texts):
        st.markdown(f"**{title}**")
        st.caption(txt)
        st.markdown("---")

# PCA & KLUSTER (visas längst ner om checkbox är markerad)
if show_pca_cluster:
    st.markdown("---")
    st.subheader("PCA-visualisering & kluster")

    plot_df = movies.copy()
    plot_df['x'] = coords[:, 0]
    plot_df['y'] = coords[:, 1]

    cluster_filter = st.multiselect("Välj kluster att visa (tom = alla)", options=sorted(plot_df['cluster'].unique().tolist()), default=[])
    if cluster_filter:
        display_df = plot_df[plot_df['cluster'].isin(cluster_filter)]
    else:
        display_df = plot_df

    rec_set = set(recs_df['Title'].tolist())
    display_df['marker'] = display_df['Title'].apply(
        lambda t: 'Vald' if t == selected_title else ('Rekommendation' if t in rec_set else 'Annat')
    )

    fig = px.scatter(
        display_df,
        x='x', y='y',
        color='cluster',
        hover_data=['Title', 'Release_Year', 'Category', 'Minutes'],
        custom_data=['Title'],
        labels={'x': 'PCA 1', 'y': 'PCA 2'},
        title=f"PCA-projicerade filmer (N={len(display_df)})"
    )

    sel_row = display_df[display_df['Title'] == selected_title]
    if not sel_row.empty:
        fig.add_trace(go.Scatter(
            x=sel_row['x'], y=sel_row['y'],
            mode='markers+text',
            marker=dict(size=20, color='red', symbol='diamond'),
            name='Vald film',
            text=sel_row['Title'],
            textposition='top center',
            hoverinfo='skip'
        ))
    rec_rows = display_df[display_df['Title'].isin(rec_set)]
    if not rec_rows.empty:
        fig.add_trace(go.Scatter(
            x=rec_rows['x'], y=rec_rows['y'],
            mode='markers',
            marker=dict(size=14, color='green', symbol='star'),
            name='Rekommendationer',
            hoverinfo='skip'
        ))

    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Utforska kluster**")
    chosen_cluster = st.selectbox("Visa filmer i kluster:", options=sorted(movies['cluster'].unique().tolist()))
    cluster_df = movies[movies['cluster'] == chosen_cluster][['Title', 'Release_Year', 'Category', 'Minutes']].reset_index(drop=True)
    st.dataframe(cluster_df)

    csv = cluster_df.to_csv(index=False).encode('utf-8')
    st.download_button("Ladda ned kluster som CSV", data=csv, file_name=f"cluster_{chosen_cluster}.csv", mime="text/csv")

# AUTOCOMPLETE UI
st.markdown("---")
st.subheader("Generera / autocompleta rekommendationstext (manuellt)")
inp_rec = st.text_input("Skriv en startfras (t.ex. 'Om du gillar Stranger Things,')", value=f"Om du gillar {selected_title},")
btn = st.button("Generera färdig text med AI" if OPENAI_KEY else "Generera template-text")
if btn:
    if OPENAI_KEY:
        try:
            resp = openai.Completion.create(
                model="text-davinci-003",
                prompt=(
                    f"Skriv en kort reklamliknande rekommendationstext på svenska utifrån startfrasen:\n\n{inp_rec}\n\n"
                    f"Max 25 ord. En mening."
                ),
                max_tokens=50,
                temperature=0.8
            )
            final_text = resp.choices[0].text.strip()
        except Exception:
            final_text = inp_rec + " ... prova också " + (rec_titles[0] if recs_df.shape[0] > 0 else "liknande titlar")
    else:
        final_text = inp_rec + " prova också " + (rec_titles[0] if recs_df.shape[0] > 0 else "liknande titlar")
    st.success(final_text)

st.markdown("### Tips")
st.write("- För bästa AI-text: sätt `OPENAI_API_KEY` som miljövariabel och bocka i rutan 'Använd OpenAI' i sidomenyn.")
st.write("- Justera antal kluster och antal rekommendationer från sidomenyn.")