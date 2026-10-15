import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from difflib import get_close_matches
import requests

import streamlit as st
from login_page import login

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
    st.stop()  # Prevent rest of app from loading

with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown('<h1 class="title">📚 Book Recommender System</h1>',unsafe_allow_html=True)

@st.cache_data
def load_data():
    books = pd.read_csv("books.csv")
    users = pd.read_csv("users.csv")
    ratings = pd.read_csv("ratings.csv")

    books['Book-Title'] = books['Book-Title'].fillna("")
    books['Book-Author'] = books['Book-Author'].fillna("")
    books['Image-URL-L'] = books['Image-URL-L'].fillna("https://via.placeholder.com/150")
    return books, users, ratings

books, users, ratings = load_data()

required_columns = {'Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L'}
if not required_columns.issubset(books.columns):
    st.error(f"Dataset is missing required columns: {required_columns - set(books.columns)}. Please provide valid datasets.")
    st.stop()

@st.cache_data
def prepare_content_model(data):
    data['description'] = data['Book-Title'] + " " + data['Book-Author']
    tfidf = TfidfVectorizer(stop_words='english', max_features=50000)
    tfidf_matrix = tfidf.fit_transform(data['description'])
    return tfidf_matrix

tfidf_matrix = prepare_content_model(books)

def get_recommendations(title, data=books, tfidf_matrix=tfidf_matrix):
    title = title.lower().strip()
    matches = get_close_matches(title, data['Book-Title'].str.lower().tolist(), n=5, cutoff=0.6)

    if not matches:
        return pd.DataFrame()

    matched_title = matches[0]
    idx = data[data['Book-Title'].str.lower() == matched_title].index[0]
    sim_scores = linear_kernel(tfidf_matrix[idx], tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-11:-1][::-1]
    return data.iloc[top_indices][['Book-Title', 'Book-Author', 'Year-Of-Publication', 'Publisher', 'Image-URL-L']]

# Function to get the book image URL
def get_image_url(url):
    if pd.isna(url) or not isinstance(url, str) or not url.startswith("http"):
        return "https://via.placeholder.com/150"
    return url

# Contact Us page function
def contact_page():
    st.title("Contact Us")
    st.markdown("Please provide your details and feedback.")

    with st.form(key='contact_form'):
        name = st.text_input("Your Name")
        email = st.text_input("Your Email")
        mobile = st.text_input("Your Mobile Number")
        feedback = st.text_area("Your Feedback")

        submit_button = st.form_submit_button(label="Submit")
        
        if submit_button:
            if name and email and mobile and feedback:
                form_data = {
                    'name': name,
                    'email': email,
                    'mobile': mobile,
                    'feedback': feedback
                }

                formspree_url = "https://formspree.io/f/mjkgorgg"

                response = requests.post(formspree_url, data=form_data)

                if response.status_code == 200:
                    st.success(f"Thank you for your feedback, {name}! We'll get back to you soon.")
                    st.session_state['page'] = 'main'  
                else:
                    st.error("Something went wrong. Please try again later.")
            else:
                st.error("Please fill in all fields.")

# Main page
def main_page():
    st.markdown("""<div class="header-section">""", unsafe_allow_html=True)
    st.markdown("### Enter a Book Title")
    book_title = st.text_input("Enter a book", placeholder="e.g., Harry Potter and the Sorcerer's Stone")
    st.markdown("""</div>""", unsafe_allow_html=True)

    if st.button("🔍 Recommend"):
        if book_title:
            with st.spinner("Fetching recommendations..."):
                recommended_books = get_recommendations(book_title)

            if not recommended_books.empty:
                st.markdown("### Recommended Books:")
                st.markdown('<div class="book-info-section">', unsafe_allow_html=True)

                for i in range(0, len(recommended_books), 3):
                    row_books = recommended_books.iloc[i:i+3] 
                    cols = st.columns(3) 
                    for col, (_, book) in zip(cols, row_books.iterrows()):
                        with col:
                            st.markdown(f"""
                                <div class="book-item">
                                    <img src="{get_image_url(book['Image-URL-L'])}" class="book-image"/>
                                    <div class="book-title">{book['Book-Title']}</div>
                                    <div class="book-info">Author: {book['Book-Author']}</div>
                                    <div class="book-info">Year: {book['Year-Of-Publication']}</div>
                                    <div class="book-info">Publisher: {book['Publisher']}</div>
                                </div>
                            """, unsafe_allow_html=True)

                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("No recommendations found. Try another book!")
        else:
            st.warning("Please enter a valid book title.")

    st.markdown("""<div class="footer">
        <p>📚 <strong>Book Recommender System</strong> | Made with ❤️ using Streamlit</p>
        <p>Designed by AVM. All rights reserved.</p>
    </div>""", unsafe_allow_html=True)

# Navigation bar
def navigation_bar():
    st.sidebar.title("Menu")
    page = st.sidebar.radio("Go to", ("Home", "Contact Us"))

    if page == "Home":
        st.session_state['page'] = 'main'
    elif page == "Contact Us":
        st.session_state['page'] = 'contact'

if 'page' not in st.session_state:
    st.session_state['page'] = 'main'

# Show navigation bar
navigation_bar()

if st.session_state['page'] == 'main':
    main_page()
elif st.session_state['page'] == 'contact':
    contact_page()
