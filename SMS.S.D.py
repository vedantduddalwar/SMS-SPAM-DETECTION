# Including all the essential modules!
import os
import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


os.chdir(os.path.dirname(os.path.abspath(__file__)))
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

# Giving Title
st.title("SMS Spam Classifier")

input_sms = st.text_area("Enter the Message")

# Button to get the output!
if st.button('Predict'):

    # 1. preprocessing the data input message
    transformed_sms = transform_text(input_sms)
    # 2. vectorizing the input message
    vector_input = tfidf.transform([transformed_sms])
    # 3. predictive analysis whether the input message is a spam or not!
    result = model.predict(vector_input)[0]
    # 4. Displaying the output if result turn out to be 1 it will be a spam and otheriwse it will be a ham!
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")

