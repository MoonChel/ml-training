import sklearn
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# url for downloading - https://www.kaggle.com/uciml/sms-spam-collection-dataset
data = pd.read_csv("spam.csv", encoding="ISO-8859-1")


# Usually axis=0 is said to be "column-wise" (and axis=1 "row-wise")
# +------------+---------+--------+
# |            |  A      |  B     |
# +------------+---------+---------
# |      0     | 0.626386| 1.52325|----axis=1----->
# +------------+---------+--------+
#              |         |
#              | axis=0  |
data = data.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

data.columns = ["labels", "sms_text"]
data["b_labels"] = data["labels"].map({"ham": 0, "spam": 1})

print(data.head(5))

# https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
cv = CountVectorizer()
X = cv.fit_transform(data["sms_text"])
Y = data["b_labels"].to_numpy()


# print(cv.get_feature_names())
# print(X.toarray())

# in this case we will get
# number of line | word 1 count | word 2 count | ... | spam or ham
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, Y_train)

train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print("train score", train_score)
print("test score", test_score)


test_sms = cv.transform(
    [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "XXXMobileMovieClub: To use your credit, click the WAP link in the",
    ]
).toarray()
prediction = model.predict(test_sms)
print(prediction)


data["predictions"] = model.predict(X)

# select sms messages where model predicted not spam, but it actually was marked as spam
sneaky_spam = data[(data["predictions"] == 0) & data["b_labels"] == 1]["sms_text"]

for sms in sneaky_spam:
    print(sms)

# select sms messages where model predicted spam, but it actually was marked as not spam
not_spam = data[(data["predictions"] == 1) & data["b_labels"] == 0]["sms_text"]

for sms in not_spam:
    print(sms)
