from preprocess_hungcuongthan import utils

__version__ = '0.0.1'

def get_wordcounts(x):
    return utils._get_wordcounts(x)

def get_charcounts(x):
    return utils.utils_get_charcounts(x)

def get_avg_wordlength(x):
    return utils._get_avg_wordlength(x)

def get_stopwords_counts(x):
    return utils._get_stopwords_counts(x)

def get_hashtag_counts(x):
    return utils._get_hashtag_counts(x)

def get_mentions_counts(x):
    return utils._get_mentions_counts(x)
    
def get_digits_counts(x):
    return utils._get_digits_counts(x)

def get_uppercase_counts(x):
    return utils._get_uppercase_counts(x)

def get_lowercase_counts(x):
    return utils._get_lowercase_counts(x)
    
def get_contraction_expansion(x):
    return utils._get_contraction_expansion(x)
    
def get_emails(x):
    return utils._get_emails(x)

def remove_emails(x):
    return utils._remove_emails(x)

def get_urls(x):
    return utils._get_urls(x)

def remove_urls(x):
    return utils._remove_urls(x)

def remove_rt(x): #remove retweet
    return utils._remove_rt(x)

def remove_special_chars(x): #remove special chars and multiple spaces
    return utils._remove_special_chars(x) #remove special chars and multiple spaces

def remove_html_tags(x):
    return utils._remove_html_tags(x)

def remove_accented_chars(x):
    return utils.remove_accented_chars(x)

def remove_stopwords(x):
    return utils._remove_stopwords(x)

def make_base(x):
    return utils._make_base(x)
    
def remove_common_words(x, n = 20):
    return utils.remove_common_words(x, n = 20)
    
def remove_rare_words(x , n = 20):
    return utils.remove_rare_words(x , n = 20)
    
def spelling_correction(x):
    return utils.spelling_correction(x)