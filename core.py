from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups

N_RESULTS = 3

class TextClassifier(object):
    """
    Wrapper class for 20Newsgroup text classifier
    """

    def __init__(self):

        self.pipeline = Pipeline([
            ('vect', TfidfVectorizer()),
            ('clf', MultinomialNB())
        ])

    def fit(self):
        """
        Fetch 20Newsgroup dataset and fit pipeline on it
    	"""

        newsgroups = fetch_20newsgroups(subset='train')
        self.target_names = newsgroups.target_names
        self.pipeline.fit(newsgroups.data, newsgroups.target)

    def predict(self, input_texts):
        """
    	Args:
      		- input_texts: ['input_str']
    	Returns:
      		- predictions: [{'target_name':score}]
    	"""

        probas = self.pipeline.predict_proba(input_texts)
        ret = []

        for proba in probas:
            ret.append(
                sorted(
                    [
                        {'label': self.target_names[i], 'score': p}
                        for i, p in enumerate(proba)
                    ],
                    key=lambda elt: elt['score'],
                    reverse=True
                )[:N_RESULTS]
            )
        return ret
