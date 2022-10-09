import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
nltk.download('punkt')
nltk.download('stopwords')


def transform(text):
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


tf = pickle.load(open('vect.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

st.title("Spam Analyzer")
input_txt = st.text_area("Enter the message")

if st.button('Predict'):
    trans_msg = transform(input_txt)
    vect = tf.transform([trans_msg])
    ans = model.predict(vect)[0]

    if ans == 1:
        st.header("It's a Spam Message")
    else:
        st.header("Not a Spam Message")
