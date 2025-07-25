# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt

# Step 2: Load Dataset
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Preprocessing
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Step 4: Text Vectorization
cv = CountVectorizer()
X = cv.fit_transform(df['message'])  # features
y = df['label_num']                 # target

# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Model Training
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Step 8: Predict a new message
test_msg = ["You’ve won a free vacation! Click to claim."]
test_vec = cv.transform(test_msg)
print("Prediction:", model.predict(test_vec))  # 1 = spam, 0 = ham
