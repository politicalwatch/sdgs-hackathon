from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk.tokenize import word_tokenize

grouped = df_test.groupby("initiative_type_alt")['content'].apply(lambda tags: ','.join(tags))

def show_cloud(i):
    text = grouped[i]

    word_tokens = word_tokenize(text)
    filtered_sentence = [w for w in word_tokens if not w.lower() in spanish_words]
    no_stop_words = ' '.join(filtered_sentence)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(no_stop_words)

    plt.figure(figsize=(12,5))

    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")

    plt.show()

    for i, row in grouped.iteritems():
        unique_id = i
        print(i)
        if len(grouped[i])>0:
            show_cloud(i)
