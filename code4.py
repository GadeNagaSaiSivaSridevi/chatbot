#Meet Robo: your friend

#import necessary libraries
import io
import random
import string # to process standard python strings
import warnings
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import nltk
from nltk.stem import WordNetLemmatizer
nltk.download('popular', quiet=True) # for downloading packages

# uncomment the following only the first time
#nltk.download('punkt') # first-time use only
#nltk.download('wordnet') # first-time use only

st.title("MEDICAL CHATBOT")

#Reading in the corpus
with open('healthcare.txt','r', errors ='ignore') as fin:
    raw = fin.read().lower()

#TOkenisation
sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences 
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# Preprocessing
lemmer = WordNetLemmatizer()
def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

a = "I am sorry! I don't understand you"
b = "ROBO: My name is Robo. I will answer your queries about Chatbots. If you want to exit, type Bye!"
c= ''
d = "ROBO:"
e = "ROBO: Bye! take care.."

# Keyword Matching
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["hi", "hey", "hi there", "hello","hi, how can i help you","i am glad! you are talking to me"]

def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


THANKS_INPUTS = ("thankyou","thanks",)
THANKS_RESPONSES = ["welcome", "Thanks welcome", "you are welcome"]

def thanks(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in THANKS_INPUTS:
            return random.choice(THANKS_RESPONSES)


# Generating response
def response(user_response):
    robo_response=''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx=vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if(req_tfidf==0):
        robo_response=robo_response+a
        return robo_response
    else:
        robo_response = robo_response+sent_tokens[idx]
        return robo_response

user_response = st.text_input(c,key=1)
flag=True
while(flag==True):
    user_response=user_response.lower()
    if(user_response!='bye'):
        if (greeting(user_response) != None):
            st.write(d + greeting(user_response))
            break
        else:
            if(thanks(user_response)!=None):
                st.write(d+thanks(user_response))
                break
            else:
                st.write(d)
                st.write(response(user_response))
                sent_tokens.remove(user_response)
                break
    else:
        flag=False
        st.write(e)