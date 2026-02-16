import pandas as pd
import re
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")

    fake["label"] = 0
    true["label"] = 1

    data = pd.concat([fake, true])
    data = data[["text", "label"]]
    data["text"] = data["text"].apply(clean_text)

    return data


def train_model(data):
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"], data["label"],
        test_size=0.2,
        random_state=42
    )

    vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_vec, y_train)
    lr_pred = lr_model.predict(X_test_vec)

    nb_model = MultinomialNB()
    nb_model.fit(X_train_vec, y_train)
    nb_pred = nb_model.predict(X_test_vec)

    lr_acc = accuracy_score(y_test, lr_pred)
    nb_acc = accuracy_score(y_test, nb_pred)

    best_model = lr_model if lr_acc > nb_acc else nb_model

    print("Logistic Regression Accuracy:", lr_acc)
    print("Naive Bayes Accuracy:", nb_acc)
    print("Best Model Selected:",
          "Logistic Regression" if best_model == lr_model else "Naive Bayes")

    joblib.dump(best_model, "fake_news_model.pkl")
    joblib.dump(vectorizer, "vectorizer.pkl")

    return best_model, vectorizer


def predict_news(model, vectorizer, text):
    text = clean_text(text)
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return "Real News" if prediction == 1 else "Fake News"


if __name__ == "__main__":
    data = load_data()
    model, vectorizer = train_model(data)

    print("\nType news to test (type 'exit' to stop)")
    while True:
        user_input = input("News: ")
        if user_input.lower() == "exit":
            break
        print("Prediction:", predict_news(model, vectorizer, user_input))
