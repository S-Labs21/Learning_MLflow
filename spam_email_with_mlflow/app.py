import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Simple_spam_classifier_model")

texts = [
    "Win money now", "Limited offer just for you",
    "Hello how are you", "Let's meet tomorrow",
    "Free gift card available", "Call me later"
]
labels = [1, 1, 0, 0, 1, 0]

X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42)

c_values = [0.1, 1.0]

for c in c_values:
    with mlflow.start_run():
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)

        model = LogisticRegression(C=c)
        model.fit(X_train_vec, y_train)

        preds = model.predict(X_test_vec)
        acc = accuracy_score(preds, y_test)

        mlflow.log_param("C", c)
        mlflow.log_metric("Accuracy", acc)

        mlflow.sklearn.log_model(model, "model")

        print(f"C={c}, Accuracy={acc}")



