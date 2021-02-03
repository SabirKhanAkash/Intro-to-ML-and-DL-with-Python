from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

catgr = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']

trainingData = fetch_20newsgroups(subset='train', categories=catgr, shuffle=True)

countVectorizer = CountVectorizer()
xTrainCounts = countVectorizer.fit_transform(trainingData.data)

tfidTransformer = TfidfTransformer()
xTrainTfidf = tfidTransformer.fit_transform(xTrainCounts)

model = MultinomialNB().fit(xTrainTfidf, trainingData.target)

new = ['This has nothing to do with religion', 'software engineering is getting hotter nowdays']
xNewCounts = countVectorizer.transform(new)
xNewTfid = tfidTransformer.transform(xNewCounts)

predicted = model.predict(xNewTfid)

for doc, catgr in zip(new,predicted):
    print('%r -------> %s' % (doc, trainingData.target_names[catgr]))
