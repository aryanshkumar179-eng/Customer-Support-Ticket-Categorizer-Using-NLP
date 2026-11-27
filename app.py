import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="Customer Support Ticket Categorizer",
    page_icon="🎫",
    layout="wide",
)


# ------------------------------------------------------------
# Sample dataset from the notebook (24 support tickets)
# ------------------------------------------------------------
TICKETS = [
    ("T001", "Billing", "I was charged twice for my monthly subscription."),
    ("T002", "Billing", "My payment failed but money was deducted."),
    ("T003", "Billing", "I need a refund for an incorrect payment."),
    ("T004", "Billing", "The amount on my invoice is incorrect."),
    ("T005", "Billing", "How can I update my credit card details?"),
    ("T006", "Billing", "My subscription renewal was charged incorrectly."),
    ("T007", "Login", "I forgot my password and cannot log in."),
    ("T008", "Login", "The password reset link is not working."),
    ("T009", "Login", "I cannot sign in after changing my password."),
    ("T010", "Login", "My account is locked and I cannot access it."),
    ("T011", "Login", "I am not receiving OTP to sign in."),
    ("T012", "Login", "I receive an error when entering my username and password."),
    ("T013", "Delivery", "My order has not arrived and delivery is late."),
    ("T014", "Delivery", "The tracking status of my package is not updating."),
    ("T015", "Delivery", "I received the wrong item in my parcel."),
    ("T016", "Delivery", "My package is marked delivered but I did not receive it."),
    ("T017", "Delivery", "Please change the delivery address for my order."),
    ("T018", "Delivery", "The courier has not delivered my order yet."),
    ("T019", "Technical", "The mobile application crashes when I open it."),
    ("T020", "Technical", "The website shows an error page."),
    ("T021", "Technical", "I cannot upload files because the system freezes."),
    ("T022", "Technical", "The screen is blank after the software update."),
    ("T023", "Technical", "The application is very slow and keeps freezing."),
    ("T024", "Technical", "I cannot connect to the service because of a network error."),
]

DEFAULT_THRESHOLD = 0.15


@st.cache_resource(show_spinner=False)
def build_text_model():
    """Fit the TF-IDF model once and reuse it between Streamlit reruns."""
    descriptions = [ticket[2] for ticket in TICKETS]
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
    )
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    return vectorizer, tfidf_matrix


def categorize_ticket(complaint: str, threshold: float):
    """Return the best category and the three most similar past tickets."""
    vectorizer, tfidf_matrix = build_text_model()
    new_vector = vectorizer.transform([complaint])
    scores = cosine_similarity(new_vector, tfidf_matrix)[0]

    top_indices = scores.argsort()[::-1][:3]
    best_index = int(top_indices[0])
    best_score = float(scores[best_index])

    matches = []
    for index in top_indices:
        ticket_id, category, description = TICKETS[int(index)]
        matches.append(
            {
                "Ticket ID": ticket_id,
                "Category": category,
                "Description": description,
                "Similarity": float(scores[int(index)]),
                "Similarity Score": f"{scores[int(index)] * 100:.2f}%",
            }
        )

    result = {
        "category": TICKETS[best_index][1] if best_score >= threshold else None,
        "ticket_id": TICKETS[best_index][0],
        "score": best_score,
        "matches": matches,
    }
    return result


# ------------------------------------------------------------
# Streamlit interface
# ------------------------------------------------------------
st.title("🎫 Customer Support Ticket Categorizer")
st.write(
    "Enter a customer complaint to find the most relevant support category "
    "and similar previously resolved tickets."
)

with st.sidebar:
    st.header("Model settings")
    threshold = st.slider(
        "Confidence threshold",
        min_value=0.00,
        max_value=1.00,
        value=DEFAULT_THRESHOLD,
        step=0.05,
        help=(
            "The app suggests a category only when the best cosine-similarity "
            "score meets or exceeds this value. The notebook's default is 0.15."
        ),
    )
    st.caption("Text features: TF-IDF with unigrams and bigrams")
    st.caption("Matching method: cosine similarity")

    st.divider()
    st.subheader("Sample dataset")
    st.metric("Total tickets", len(TICKETS))
    category_counts = pd.Series([ticket[1] for ticket in TICKETS]).value_counts()
    for category, count in category_counts.items():
        st.write(f"**{category}:** {count}")

complaint = st.text_area(
    "Customer complaint",
    placeholder="Example: I was charged twice for my subscription...",
    height=130,
)

analyze = st.button("Analyze complaint", type="primary", use_container_width=True)

if analyze:
    cleaned_complaint = complaint.strip()

    if not cleaned_complaint:
        st.warning("Please enter a customer complaint before analyzing.")
    else:
        with st.spinner("Comparing your complaint with past tickets..."):
            result = categorize_ticket(cleaned_complaint, threshold)

        st.subheader("Categorization result")
        left, right = st.columns(2)

        with left:
            if result["category"] is not None:
                st.success(f"Suggested category: **{result['category']}**")
            else:
                st.warning(
                    "Low confidence — no reliable category could be suggested. "
                    "Please review the complaint manually."
                )

        with right:
            st.metric(
                "Best similarity score",
                f"{result['score'] * 100:.2f}%",
                help="Cosine similarity between the new complaint and the closest past ticket.",
            )
            st.progress(min(max(result["score"], 0.0), 1.0))

        st.write(f"**Closest past ticket:** {result['ticket_id']}")

        st.subheader("Top 3 similar past tickets")
        display_matches = pd.DataFrame(result["matches"])[
            ["Ticket ID", "Category", "Description", "Similarity Score"]
        ]
        st.dataframe(display_matches, use_container_width=True, hide_index=True)

        with st.expander("How does this app work?"):
            st.markdown(
                """
                1. The sample ticket descriptions are converted into TF-IDF vectors.
                2. Your complaint is converted using the same fitted vectorizer.
                3. Cosine similarity compares your complaint with every sample ticket.
                4. The app shows the three closest tickets.
                5. The closest ticket's category is suggested only when its score
                   meets the selected confidence threshold.
                """
            )

with st.expander("View all sample support tickets"):
    all_tickets = pd.DataFrame(
        TICKETS, columns=["Ticket ID", "Category", "Description"]
    )
    st.dataframe(all_tickets, use_container_width=True, hide_index=True)

st.caption("Demo app using the 24 sample tickets from the supplied Jupyter notebook.")
