import string
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_fscore_support as score

stopword = stopwords.words('english')
ps = nltk.PorterStemmer()
data_processed = pd.read_csv("Tweets.csv")

# Data cleaning
features = data_processed.iloc[:, 10].values
labels = data_processed.iloc[:, 1].values

processed_features = []

for sentence in range(0, len(features)):
    # Remove all the special characters
    processed_feature = re.sub(r'\W', ' ', str(features[sentence]))

    # remove all single characters
    processed_feature= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_feature)

    # Remove single characters from the start
    processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_feature)

    # Substituting multiple spaces with single space
    processed_feature = re.sub(r'\s+', ' ', processed_feature, flags=re.I)

    # Removing prefixed 'b'
    processed_feature = re.sub(r'^b\s+', '', processed_feature)

    # Converting to Lowercase
    processed_feature = processed_feature.lower()

    processed_features.append(processed_feature)

print(len(processed_features))

processed_features = []
def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100

def clean(text):
    text = "".join([word for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopword]
    return text

for sentence in range(0, len(features)):
    txt = clean(features[sentence])
    processed_features.append(" ".join(i.lower() for i in txt))

tfidk_vect = TfidfVectorizer(analyzer=clean)
X_tfid_features = tfidk_vect.fit_transform(processed_features).toarray()
X_train, X_test, Y_train, Y_test = train_test_split(X_tfid_features,labels, test_size=0.2)

"""
Using RandomForest
"""
txt_classifier = RandomForestClassifier(n_estimators=200,random_state=0)
txt_classifier.fit(X_train,Y_train)

predictions = txt_classifier.predict(X_test)
print("-----------------For RandomForest ---------------------")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(accuracy_score(Y_test, predictions))

# Accuracy of 0.7568306010928961
"""
Evaluate using GridSearchCV
"""
txt_classifier = GradientBoostingClassifier()
# print(txt_classifier.get_params().keys())
parameters = {"n_estimators":[100,150],"max_depth":[7,11,15],"learning_rate":[0.1]}
gs = GridSearchCV(txt_classifier,parameters,cv=5,n_jobs=-1)
gs_fit = gs.fit(X_train,Y_train)
predictions = gs_fit.predict(X_test)
print("\n-----------------For GridSearch ---------------------")
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(accuracy_score(Y_test, predictions))


