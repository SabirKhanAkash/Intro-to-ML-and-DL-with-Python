import _collections
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

def tokenizer(text):
    tokens = word_tokenize(text)
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens if t not in stopwords.words('english')]
    return tokens

def clusterSentences(sentences, nb_of_clusters=3):
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenizer, stop_words=stopwords.words('english'),lowercase=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    kmeans = KMeans(n_clusters=nb_of_clusters)
    kmeans.fit(tfidf_matrix)
    clusters = _collections.defaultdict(list)
    for i, label in enumerate(kmeans.labels_):
        clusters[label].append(i)
    return dict(clusters)

if __name__ == "__main__":
    sentences = ["Quantum physics is quite important in science nowdays.",
                        "Software engineering is hotter and hotter topic in the silicon valley",
                        "Investing in stocks and trading with them are not that easy",
                        "FOREX is the stock market for trading currencies",
                        "Warren Buffet is famous for making good investments. He knows stock markets",
                        "Hasi-b is working on video editing and earning foreign currencies",
                        "Akash is trying to learn something new and trying to contribute something in the GitHub profile on a daily basis but will he able to do it?"
                       ]
    nclusters = 3
    clusters = clusterSentences(sentences,nclusters)
    for cluster in range(nclusters):
        print("\nCLUSTER ",cluster,":")
        for i, sentence in enumerate(clusters[cluster]):
            print("\tSENTENCES ",i,": ",sentences[sentence])
