import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(page_title="News Article Topic Finder", page_icon="📰", layout="centered")

# Step 1: Sample dataset of 20 news articles
articles = [
    ("A001", "Electric Vehicles Revolution",
     "New electric vehicles are changing the automobile industry with advanced battery technology."),
    ("A002", "Battery Technology Advances",
     "Scientists develop powerful batteries that charge faster and last longer."),
    ("A003", "Future of Electric Cars",
     "Electric car companies introduce new models with improved driving range."),
    ("A004", "Solar Energy Development",
     "Renewable energy companies invest in solar power and clean energy technology."),
    ("A005", "Artificial Intelligence Growth",
     "Artificial intelligence is transforming businesses through automation and machine learning."),
    ("A006", "New Smartphone Technology",
     "Technology companies launch smartphones with advanced processors and improved cameras."),
    ("A007", "Indian Cricket Team Victory",
     "India wins an international cricket match after an impressive batting performance."),
    ("A008", "Football Championship News",
     "Football teams prepare for the upcoming international championship."),
    ("A009", "Stock Market Updates",
     "The stock market rises as investors increase their investments in major companies."),
    ("A010", "Startup Investment Growth",
     "Indian startups receive funding from investors to expand their businesses."),
    ("A011", "New Government Policies",
     "The government announces new policies to improve economic development."),
    ("A012", "Election Campaign Updates",
     "Political parties begin their campaigns before the upcoming national elections."),
    ("A013", "Bollywood Movie Release",
     "A popular Bollywood actor announces a new movie scheduled for release."),
    ("A014", "Music Industry Trends",
     "Popular singers release new music albums and announce upcoming concerts."),
    ("A015", "Cybersecurity Developments",
     "Technology experts introduce new cybersecurity systems to protect online data."),
    ("A016", "Electric Vehicle Charging",
     "New charging stations help electric vehicle owners charge their cars quickly."),
    ("A017", "Advanced Lithium Batteries",
     "Researchers develop lithium battery technology for electric cars and energy storage."),
    ("A018", "Online Shopping Business",
     "E-commerce companies introduce new services to improve online shopping experiences."),
    ("A019", "Sports Tournament Results",
     "Athletes compete in international sports tournaments and win medals."),
    ("A020", "Space Technology Research",
     "Scientists develop new space technology for future satellite missions.")
]

# Step 2: Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Step 3: Cache the TF-IDF model so it doesn't recompute on every click
@st.cache_resource
def build_vectorizer(article_list):
    documents = [preprocess_text(a[1] + " " + a[2]) for a in article_list]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_vectorizer(articles)

# --- STREAMLIT UI ---
st.title("📰 News Article Topic Finder")
st.write("Search for related news articles using **TF-IDF** and **Cosine Similarity**.")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    top_n = st.slider("Number of results to display", min_value=1, max_value=10, value=5)
    st.divider()
    st.caption(f"Total articles indexed: {len(articles)}")

# Search input form
with st.form("search_form"):
    query = st.text_input("Enter your news topic or headline:", placeholder="e.g., electric cars, cricket, AI...")
    submitted = st.form_submit_button("Find Related Articles")

# Search execution and results display
if submitted:
    cleaned_query = preprocess_text(query)

    if not cleaned_query:
        st.warning("Please enter a valid news topic using alphabetic keywords.")
    else:
        query_vector = vectorizer.transform([cleaned_query])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix)[0]
        top_indices = similarity_scores.argsort()[::-1][:top_n]

        st.subheader(f"Top {top_n} Related News Articles")

        # Check if the highest score is 0 (no matching keywords found)
        if similarity_scores[top_indices[0]] == 0:
            st.info("No matching articles found for your search topic. Try different keywords like 'battery', 'sports', or 'technology'.")
        else:
            for rank, index in enumerate(top_indices, start=1):
                score_pct = round(similarity_scores[index] * 100, 2)
                article_id, title, summary = articles[index]

                with st.container(border=True):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.markdown(f"**#{rank} — {title}** (`{article_id}`)")
                        st.write(summary)
                    with col2:
                        st.metric(label="Relevance", value=f"{score_pct}%")
