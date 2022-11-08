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




# ---------------------------------------------------------
# Class: Text Processing / Normalizing 
# ---------------------------------------------------------
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
        
        
        
        
        
# ---------------------------------------------------------
# Function: Model evaluation  
# ---------------------------------------------------------        
def model_Evaluate_values(model, x_train, x_test, y_train, y_test, model_name, balanced=True):
    """
    """
    # Print accuracy scores on train and test sets
    R_train = model.score(x_train, y_train)
    R_test  = model.score(x_test, y_test)
    df_accuracy = pd.DataFrame( np.round( [R_train, R_test], 2 ),  columns=['score'])
    
    df_accuracy['metric'] = ['R_train' , 'R_test']
    df_accuracy['model'] = model_name
    if balanced == True:
        df_accuracy['balanced'] = 'yes'
    if balanced == False:
        df_accuracy['balanced'] = 'no'
        
    # Predict values for Test dataset
    y_pred = model.predict(x_test)
    scores = precision_recall_fscore_support(y_test, model.predict(x_test))
    df_precision_recall = pd.DataFrame( np.round(scores, 2) , columns=['is_pandemicPreps', 'is_covid19positive'] )
    df_precision_recall['metric'] = ['precision' , 'recall' , 'fscore' , 'support']
    df_precision_recall['model'] = model_name
    if balanced == True:
        df_precision_recall['balanced'] = 'yes'
    if balanced == False:
        df_precision_recall['balanced'] = 'no'

    
    cf_matrix = confusion_matrix(y_test, y_pred)
    categories = ['Negative','Positive']
    group_names = ['True Neg','False Pos', 'False Neg','True Pos']
    group_percentages = [np.round(value, 2) for value in cf_matrix.flatten() / np.sum(cf_matrix)]
    df_cf_matrix = pd.DataFrame( group_percentages ,  columns=['score'] )
    df_cf_matrix['metric'] = group_names
    df_cf_matrix['model'] = model_name
    if balanced == True:
        df_cf_matrix['balanced'] = 'yes'
    if balanced == False:
        df_cf_matrix['balanced'] = 'no'
        
    # save_______________________________
    # if file is not there
#     pd.DataFrame(df_accuracy).to_csv('../datasets/models_metrics_report_accuracy.csv')
#     pd.DataFrame(df_cf_matrix).to_csv('../datasets/models_metrics_report_confusionMatrix.csv')
#     pd.DataFrame(df_precision_recall).to_csv('../datasets/models_metrics_report_precision_recall.csv')
    # accuracy - 1
    df1 = pd.read_csv('../datasets/models_metrics_report_accuracy.csv', index_col=0)
    df2 = df_accuracy
    pd.concat([df1,df2],ignore_index=True).to_csv('../datasets/models_metrics_report_accuracy.csv')
    # accuracy - 0
    df1 = pd.read_csv('../datasets/models_metrics_report_precision_recall.csv', index_col=0)
    df2 = df_precision_recall
    pd.concat([df1,df2],ignore_index=True).to_csv('../datasets/models_metrics_report_precision_recall.csv')
    # accuracy - 2
    df1 = pd.read_csv('../datasets/models_metrics_report_confusionMatrix.csv', index_col=0)
    df2 = df_cf_matrix
    pd.concat([df1,df2],ignore_index=True).to_csv('../datasets/models_metrics_report_confusionMatrix.csv')
    
    #     print(classification_report(y_test, y_pred))
    return df_accuracy, df_precision_recall, df_cf_matrix        