# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect File Encoding: Use chardet to determine the dataset's encoding.
2. Load Data: Read the dataset with pandas.read_csv using the detected encoding.
3. Inspect Data: Check dataset structure with .info() and missing values with .isnull().sum().
4. Split Data: Extract text (x) and labels (y) and split into training and test sets using train_test_split.
5. Convert Text to Numerical Data: Use CountVectorizer to transform text into a sparse matrix.
6. Train SVM Model: Fit an SVC model on the training data.
7. Predict Labels: Predict test labels using the trained SVM model.
8. Evaluate Model: Calculate and display accuracy with metrics.accuracy_score.

## Program:
```
Program to implement the SVM For Spam Mail Detection.
Developed by: BARATH V
RegisterNumber: 212225240023
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 2: Load the dataset
data = pd.read_csv("spam.csv", encoding='latin-1')

# Step 3: Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'message']



# Step 4: Convert labels to numeric
# ham = 0, spam = 1
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# Step 5: Separate features and target
X = data['message']
y = data['label_num']

# Step 6: Convert text to numerical features using TF-IDF Term Frequency–Inverse Document Frequency
vectorizer = TfidfVectorizer(stop_words='english') #remove common words like is, are
X_tfidf = vectorizer.fit_transform(X)


# Step 7: Split into Training and Testing data
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

# Step 8: Train SVM Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)

# Step 9: Predict on Test Data
y_pred = svm_model.predict(X_test)
user_message = input("\nEnter a message to check whether it is Spam or Ham:\n")

# Convert user message into TF-IDF features
user_tfidf = vectorizer.transform([user_message])

# Predict using trained SVM model
prediction = svm_model.predict(user_tfidf)

# Display result
if prediction[0] == 1:
    print("\n This message is SPAM")
else:
    print("\n This message is HAM (Not Spam)")
# Step 10: Evaluate the Model
accuracy = accuracy_score(y_test, y_pred)

print("\n🔹 SVM Spam Mail Detection Results")
print("Accuracy:", accuracy)
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

## Output:
<img width="637" height="495" alt="image" src="https://github.com/user-attachments/assets/7ab0cc85-df39-40f3-9239-b38833c8da43" />









## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
