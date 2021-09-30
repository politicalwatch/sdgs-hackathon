# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Import libraries

# %% [markdown]
# ## TODO: move to requirements.txt

# %%
# !pip install stop-words
# !pip install wordcloud
# !pip install stanza
# !pip install spacy-stanza

# %%
import pandas as pd
import numpy as np
import re
import unidecode
from nltk.probability import FreqDist
from nltk.corpus import stopwords as swords
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import stanza
import spacy_stanza
import itertools
import nltk
nltk.download('stopwords')
# Download the stanza model if necessary
stanza.download("es")

# Initialize the pipeline
nlp = spacy_stanza.load_pipeline("es")

# %% [markdown]
# # Load the dataset

# %%
df = pd.read_pickle('initiatives.pkl') 

# %%
df.head()

# %%
# If content is not present, use the title as content
df['content_coalesce'] = df['content'].combine_first(df['title'])

# %%
# test
df[df['content'].isna()][['content_coalesce','title','content']]

# %%
# Here we flatten and then create lists of the individual words
# df['lists_content_coalesce'] = [''.join(l).split(" ") for l in df['content_coalesce']]

# %%
# THIS FLATTENS
df['content_coalesce'] = [''.join(l) for l in df['content_coalesce']]


# %% [markdown]
# # Define helper functions

# %%
def create_df_from_json(file_path, columns_to_keep=['content','title','initiative_type_alt'],field_name='initiatives'):
    with open(file_path,'r', encoding="utf8") as f:
        data = json.loads(f.read())
    data_frame = pd.json_normalize(data,record_path=field_name)
    data_frame = data_frame[columns_to_keep]
    return(data_frame)

def retrieve_stop_words():
    spanish_stopwords = swords.words('spanish')
    stop_words_spanish = get_stop_words('spanish')
    stopwords = list(set(spanish_stopwords + stop_words_spanish))
    return stopwords

def space_out_your_text(row):
    doc = nlp(row)
    cleaned = ""
    for token in doc:
        if token.pos_ not in ("PUNCT","ADP","SCONJ","PRON","CCONJ"):
            #print(token.text, token.lemma_, token.pos_, token.dep_)
            cleaned+=token.lemma_+" "
    return cleaned


def remove_accents(row,column):
    return unidecode.unidecode(row[column])

#remove special characters
def replace_special_char(row):
    for word, initial in {".":" ", "-":" ","/":" ","@":" ","#":" ","(":" ",")":" ",'"' : ""}.items():
        row = row.replace(word, initial) 
    return row

def remove_stopwords(row, stopwords):
    removed_stopwords = " ".join([word for word in row.split(" ") if word not in stopwords and word.replace(" ","")!=""])
    return removed_stopwords


def remove_numbers(col):
    return col.str.replace('\d+', '')


def unique_words(col):
    words = col.str.lower().str.findall("\w+")
    unique = set()

    for x in words:
        unique.update(x)
    return unique


def word_count(df):
    tf = df['text'].apply(lambda x: FreqDist(x)).sum(axis = 0)
    tf2 = dict(tf)
    data_items = tf2.items()
    data_list = list(data_items)
    freq_dataframe = pd.DataFrame(data_list)
    freq_dataframe.columns = ['Word','Counts']
    freq_dataframe = freq_dataframe.sort_values(by="Counts",ascending=False)
    pd.set_option("max_rows", None)
    return freq_dataframe


# %% [markdown]
# # Get stopwords

# %%
stop_words = retrieve_stop_words()

# %% [markdown]
# # Demo Wordcloud

# %%
from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# example usage: test_df = create_df_from_json('./small-batch.json')
test_df = create_df_from_json('small-batch.json') 

grouped = test_df.groupby("initiative_type_alt")['content'].apply(lambda tags: ','.join(tags))

def show_cloud(i):
    text = grouped[i]
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    plt.figure(figsize=(12,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

for i, row in grouped.iteritems():
    unique_id = i   
    print(i)    
    if len(grouped[i])>0:
        show_cloud(i) 

# %% [markdown]
# # Apply transformations one-by-one

# %%
unique = unique_words(df['content_coalesce'])
print("Nr of words: " + str(len(list(unique))))

# %%
df['t1_no_accents'] = df.apply(lambda row:remove_accents(row,'content_coalesce'),axis=1)
df['t1_no_accents'].head()

# %%
df['t2_no_numbers'] = remove_numbers(df['t1_no_accents']) 
df['t2_no_numbers'].head()

# %%
df['t3_no_special_char'] = df['t2_no_numbers'].apply(lambda row:replace_special_char(row))
df['t3_no_special_char'].head()

# %%
df['t4_lowercase'] = df['t3_no_special_char'].str.lower()
df['t4_lowercase'].head()

# %%
df["unigrams"] = df["t4_lowercase"].apply(nltk.word_tokenize)

# %%
df["unigrams"].head()

# %%
df['t5_stopwords_removed'] = df['t4_lowercase'].apply(lambda row:remove_stopwords(row, stop_words))
df['t5_stopwords_removed'].head()


# %%
def lemmatize(text):
    return " ".join([tok.lemma_ for tok in nlp.tokenizer(text) if not tok.is_stop])

df['t6_lemmitization'] = df['t5_stopwords_removed'].apply(lambda x:lemmatize(x))
df['t6_lemmitization'].head()

# %%
df['t5_stopwords_removed'].head()

# %%
unique = unique_words(df['t3_no_special_char'])
print("Nr of words: " + str(len(list(unique))))

# %%
unique = unique_words(df['t5_stopwords_removed'])
print("Nr of words: " + str(len(list(unique))))

# %%
unique = unique_words(df['t6_lemmitization'])
print("Nr of words: " + str(len(list(unique))))

# %%

# %%
df[df._id == "8139ec1c206e10ba04dde86d3e06f2698e34b0a6"]['content_coalesce'].values

# %%
# def transform(df):
#     df['content_coalesce'] = df.apply(lambda row:remove_accents(row,'content_coalesce'),axis=1)
# a = transform(df)
# a['content_coalesce'].head()

# %%

# %%

# %%

# %%

# %%
#remove punctuations, tabs, etc 
df.apply(lambda row:space_out_your_text(row['lowered']),axis=1)
#Lower case
df.apply(lambda row: row['text'].lower(), axis=1)
df['removed_num'] = df.apply(lambda row: remove_numbers(row['text']), axis=1)    

# %%
df2['tokenized_sents'] = df2.apply(lambda row: nlp(row['removed_num']), axis=1)
df2['tokenized_sents_str'] = df2.tokenized_sents.apply(lambda x:str(x))

infreq = freq_dataframe[freq_dataframe['Counts'] < 3 ]['Word'].tolist()


df2['removed_infreq'] = df2.tokenized_sents_str.apply(lambda x: remove_stopwords(x,infreq))
df2['removed_infreq_str'] = df2.removed_infreq.apply(lambda x:str(x))



v = TfidfVectorizer()
x = v.fit_transform(df2['removed_infreq_str'])
x.todense()
len(v.vocabulary_)
