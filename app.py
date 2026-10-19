
import joblib
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Spam Message Classifier",
    page_icon="📩",
    layout="centered"
)

# Load the trained model and vectorizer
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load("spam_classifier.joblib")
    return artifacts["vectorizer"], artifacts["model"]

try:
    vectorizer, model = load_artifacts()
except Exception:
    st.error(
        "Could not load the model. "
        "Ensure spam_classifier.joblib is in the project folder."
    )
    st.stop()

# Header
st.title("📩 Spam Message Classifier")
st.write(
    "Enter a text message below to check whether "
    "it is classified as spam or ham."
)

st.info(
    "This application uses traditional NLP, "
    "Bag of Words features, and Multinomial Naive Bayes."
)

# Message input
message = st.text_area(
    "Enter your message",
    placeholder="Example: Congratulations! Claim your free prize now...",
    height=150
)

# Prediction
if st.button("Analyze Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message first.")
    else:
        message_vector = vectorizer.transform([message])
        prediction = int(model.predict(message_vector)[0])
        probabilities = model.predict_proba(message_vector)[0]

        ham_probability = float(probabilities[0])
        spam_probability = float(probabilities[1])

        st.divider()
        st.subheader("Classification Result")

        if prediction == 1:
            st.error("🚨 This message is classified as SPAM")
        else:
            st.success("✅ This message is classified as HAM")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Spam probability",
                f"{spam_probability:.2%}"
            )

        with col2:
            st.metric(
                "Ham probability",
                f"{ham_probability:.2%}"
            )

        st.caption(
            "Probabilities are model estimates, not guarantees. "
            "Some messages may be misclassified."
        )

# Footer
st.divider()
st.caption("Built with Python, Scikit-learn and Streamlit.")
