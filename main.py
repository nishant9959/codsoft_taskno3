import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Dataset ---
# Load the data from the provided CSV file into a pandas DataFrame.
# The file is expected to be in the same directory as this script.
try:
    iris_df = pd.read_csv('IRIS.csv')
    print("Dataset loaded successfully!")
    print("First 5 rows of the dataset:")
    print(iris_df.head())
    print("\n" + "="*50 + "\n")
except FileNotFoundError:
    print("Error: IRIS.csv not found. Please make sure the file is in the correct directory.")
    exit()

# --- 2. Data Preprocessing ---
# Separate the features (X) from the target variable (y).
# Features are the measurements of the flowers.
X = iris_df.drop('species', axis=1)
# Target is the species of the flower.
y = iris_df['species']

# Encode the categorical target variable 'species' into numerical labels.
# Machine learning models require numerical input.
# Iris-setosa -> 0, Iris-versicolor -> 1, Iris-virginica -> 2
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
species_names = label_encoder.classes_

print("Features (X) shape:", X.shape)
print("Target (y) shape:", y_encoded.shape)
print("Species will be encoded as follows:")
for i, name in enumerate(species_names):
    print(f"{name} -> {i}")
print("\n" + "="*50 + "\n")


# --- 3. Split Data into Training and Testing Sets ---
# We split the data to train the model on one part and test its performance on another, unseen part.
# test_size=0.2 means 20% of the data will be used for testing.
# random_state ensures that the split is the same every time we run the code.
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("\n" + "="*50 + "\n")


# --- 4. Train the Classification Model ---
# We will use Logistic Regression, a simple yet effective classification algorithm.
# We create an instance of the model and train it using the .fit() method on our training data.
print("Training the Logistic Regression model...")
model = LogisticRegression(max_iter=200) # Increased max_iter for convergence
model.fit(X_train, y_train)
print("Model training complete!")
print("\n" + "="*50 + "\n")


# --- 5. Evaluate the Model ---
# Now we use the trained model to make predictions on the test set.
print("Evaluating the model on the test set...")
y_pred = model.predict(X_test)

# Calculate the accuracy of the model.
# Accuracy is the proportion of correctly predicted instances.
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\n")

# Display a detailed classification report.
# This shows precision, recall, and f1-score for each class.
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=species_names))
print("\n" + "="*50 + "\n")


# --- 6. Visualize the Results (Confusion Matrix) ---
# A confusion matrix is a great way to visualize the performance of a classification model.
# The diagonal elements show the number of correct predictions for each class.
print("Generating Confusion Matrix...")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=species_names, yticklabels=species_names)
plt.title('Confusion Matrix')
plt.ylabel('Actual Species')
plt.xlabel('Predicted Species')
plt.show()

print("Script finished.")

