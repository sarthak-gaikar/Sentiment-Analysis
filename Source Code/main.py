import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# Load Data
twitter_training = pd.read_csv('twitter_training.csv')

# Data Preprocessing
# Dropping some unnecessary columns
twitter_training.dropna(inplace=True)
twitter_training = twitter_training.drop('Id', axis=1)

# Checking for missing values
# print(twitter_training.isnull().sum())

# Checking duplicate values
# print(twitter_training.duplicated().sum())
twitter_training = twitter_training.drop_duplicates()
# print(twitter_training.duplicated().sum())

# Visualizing Data
colors = ['orange', 'green', 'blue', 'gray']

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Pie chart of sentiment distribution
axs[0].pie(twitter_training['sentiment'].value_counts(), labels=twitter_training['sentiment'].unique(), autopct='%1.1f%%',
            startangle=90, wedgeprops={'linewidth': 0.5}, textprops={'fontsize': 12},
            explode=[0.05, 0.05, 0.05, 0.05], colors=colors, shadow=False)
axs[0].set_title('Sentiment Distribution - Pie Chart')

# Bar chart for sentiment distribution
axs[1] = twitter_training['sentiment'].value_counts().plot(kind='bar', color=colors, ax=axs[1])
axs[1].set_title('Sentiment Distribution - Bar Plot')
axs[1].set_xlabel('Sentiment')
axs[1].set_ylabel('Count')
axs[1].tick_params(axis='x', rotation=0)
axs[1].grid(axis='y', linestyle='--', alpha=0.7)

# Adding label
for p in axs[1].patches:
    axs[1].annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='center', xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.show()

# Cross-tabulation table
plt.figure(figsize=(10, 6))
count_table = pd.crosstab(index=twitter_training['branch'], columns=twitter_training['sentiment'])
sns.heatmap(count_table, cmap='YlOrRd', annot=True, fmt='d', linewidths=0.5, linecolor='black')
plt.title('Sentiment Distribution by Branch')
plt.xlabel('Sentiment')
plt.ylabel('Branch')
plt.show()
   
def vectorize_data(text_data):
    # Join the tokenized text into strings
    text_data_strings = [" ".join(tokens) for tokens in text_data]
    # Initialize TfidfVectorizer
    tfidf_vectorizer = TfidfVectorizer()
    # Fit and transform the text data to generate TF-IDF vectors
    tfidf_vectors = tfidf_vectorizer.fit_transform(text_data_strings)
    return tfidf_vectors, tfidf_vectorizer

# Test/Validation data
# Load test data
twitter_validation = pd.read_csv('twitter_validation.csv')
twitter_validation.dropna(inplace=True)
twitter_validation = twitter_validation.drop('Id', axis=1)
twitter_validation = twitter_validation.drop_duplicates()

# Joiining Test and Validation Data
df = pd.concat([twitter_training, twitter_validation], ignore_index=False)

# Feature and Target data
X = df['tweet'] 
y = df['sentiment']  

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Random Forest classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train_tfidf, y_train)

# Predict on the testing data
y_pred = rf_classifier.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", accuracy)

# Visulizing Predictions 
# Create a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Create a classification report
class_report = classification_report(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Print classification report
print("Classification Report:")
print(class_report)

# Print some actual vs predicted labels along with tweet text
print("Actual vs Predicted Labels with Tweet Text:")
for tweet, actual_label, predicted_label in zip(X_test[:10], y_test[:10], y_pred[:10]):
    print("Tweet:", tweet)
    print("Actual Label:", actual_label)
    print("Predicted Label:", predicted_label)
    print("-----------------------")
