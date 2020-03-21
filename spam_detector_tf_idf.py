import sklearn
import pandas as pd

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

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
tf_idf = TfidfVectorizer()
X = tf_idf.fit_transform(data["sms_text"])
Y = data["b_labels"].to_numpy()

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model = MultinomialNB()
model.fit(X_train, Y_train)

train_score = model.score(X_train, Y_train)
test_score = model.score(X_test, Y_test)

print("train score", train_score)
print("test score", test_score)


test_sms = tf_idf.transform(
    [
        "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
        "XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL",
        "England v Macedonia - dont miss the goals/team news. Txt ur national team to 87077 eg ENGLAND to 87077 Try:WALES, SCOTLAND 4txt/̼1.20 POBOXox36504W45WQ 16+",
    ]
).toarray()
prediction = model.predict(test_sms)
print(prediction)
