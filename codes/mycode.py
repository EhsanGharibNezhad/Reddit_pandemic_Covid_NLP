# ---------------------------------------------------------
# Text Processing Libraries
# ---------------------------------------------------------
from bs4 import BeautifulSoup #Function for removing html
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer
import re
import demoji
import emoji
from nltk.corpus import stopwords





class text_processing():
    """
    This class includes the following functions to process the text files:
    1. read_files_in_path: 
                to read all files
    2. read_words: 
                to extract a list of words from each text file
    3. make_corpus: 
                to convert each text document into a single corpus
                Results would be a list of corpus
    4. corpus_to_info_blocks:
                Split different sections in the corpus 
                e.g., Admission Date, Sex, Discharge Date, Date of Birth...
    """
    def __init__(self, post, title):
        self.post = post
        self.title = title

    def lower_casting(self, text):
        ## Read the list of files
        self.lower_casting = text.str.lower()

        
    # Function for url's
    def remove_urls(self, text):
        self.remove_urls = text\
                        .replace('https?://\S+|www\.\S+', '', regex=True)\
        
        
    
    def remove_special_chars(self,text):
        self.remove_special_chars = text\
                        .replace('http\S+', '', regex=True)\
                        .replace('www\S+', '', regex=True)\
                        .replace('\n\n\S+', '', regex=True)\
                        .replace('\n', '', regex=True)\
                        .replace('\*', '', regex=True)\
                        .replace('!', '', regex=True)\
                        .replace("[^a-zA-Z]", ' ', regex=True) # replace_all_non_letters_with_space
        
    def count_emoji(self, text):
        self.count_emoji = (text.map(demoji.findall) != {}).sum()

    def remove_emoji(self, text):
        self.remove_emoji = text.map(demoji.replace)      
        
    def convert_emoji_to_text(self, text):
        self.convert_emoji_to_text = emoji.demojize(text) 

        
    def remove_stop_words(self, text):
        self.remove_stop_words = text.apply(stopwords)
        
        
    def remove_stopwords_fun(text):
        STOPWORDS = set(stopwords.words('english'))# Function to remove the stopwords
        return " ".join([word for word in str(text).split() if word not in STOPWORDS])# Applying the stopwords to 'text_punct' and store into 'text_stop'

    def remove_stopwords(self, text, fun = remove_stopwords_fun):
        self.remove_stopwords = text.apply(fun)    
        

    def make_token_fun(data):
        Pstemmizer = PorterStemmer()
        tokenizer = RegexpTokenizer(r'\w+') # remove the punctuation 
        post_tokens = tokenizer.tokenize(data)
        stem_spam = [Pstemmizer.stem(token) for token in post_tokens]
        return (' '.join(stem_spam))

    def stemming_text (self, 
                       text, 
                       func = make_token_fun):
        self.stemming_text = list(map(func, text))        