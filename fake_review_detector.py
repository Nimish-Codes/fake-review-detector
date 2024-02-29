import streamlit
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Load pickled model
with open('fake_review_deetct_model.pkl', 'rb') as f:
  model = pickle.load(f)

with open('fake_review_detect_vectorizer.pkl', 'rb') as f:
  vectorizer = pickle.load(f)

# Prediction function
def predict_review(model, vectorizer, text):
    text_vect = vectorizer.transform([text])
    prediction = model.predict(text_vect)
    return prediction[0]

# Example usage
user_input = st.text_area("Enter your review: ")
if st.button('Check'):
  prediction = predict_review(model, vectorizer, user_input)
  if prediction == "OR":
    st.error("Fake review")
  else:
    st.success("Genuine review")
