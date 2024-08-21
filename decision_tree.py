import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from sklearn import tree
import pydotplus
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image as PILImage


# Load the CSV file into a DataFrame
data = pd.read_csv('C:\\Users\\HP8CG\\OneDrive\\Documents\\PROJECTS\\grip\\Iris.csv')

# Explore the Data
print("First few rows of the dataset:")
print(data.head())

print("\nData info:")
print(data.info())

print("\nData description:")
print(data.describe())

# Drop the 'Id' column as it is not needed for modeling
data = data.drop('Id', axis=1)

# Encode the 'Species' column into numeric values
data['Species'] = data['Species'].astype('category').cat.codes

# Separate Features and Labels
X = data.drop('Species', axis=1)
y = data['Species']

# Split the Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and Train the Model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'\nModel Accuracy: {accuracy:.2f}')

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the Model
joblib.dump(model, 'decision_tree_model.pkl')

# Load the Model (Optional)
# model = joblib.load('decision_tree_model.pkl')


# Predicting New Data
new_data = pd.DataFrame({
    'SepalLengthCm': [5.1],
    'SepalWidthCm': [3.5],
    'PetalLengthCm': [1.4],
    'PetalWidthCm': [0.2]
})

# Predict the class for new data
predictions = model.predict(new_data)
# Convert numeric predictions back to original labels
species = data['Species'].astype('category').cat.categories
predicted_species = [species[i] for i in predictions]
print("\nPredictions for new data:")
print(predicted_species)


# Create a dot file for visualization
dot_data = tree.export_graphviz(model, out_file=None, 
                                feature_names=X.columns,  
                                class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],  
                                filled=True, rounded=True,  
                                special_characters=True)  

# Generate the graph
graph = pydotplus.graph_from_dot_data(dot_data)

# Save the graph as a PNG file
graph.write_png('decision_tree.png')

# Display the image using matplotlib
img = PILImage.open('decision_tree.png')
plt.figure(figsize=(10, 10))
plt.imshow(img)
plt.axis('off')
plt.show()

# Plot feature distributions
plt.figure(figsize=(12, 6))

# Sepal Length Distribution
plt.subplot(1, 2, 1)
sns.histplot(data['SepalLengthCm'], kde=True)
plt.title('Sepal Length Distribution')

# Petal Length Distribution
plt.subplot(1, 2, 2)
sns.histplot(data['PetalLengthCm'], kde=True)
plt.title('Petal Length Distribution')

plt.tight_layout()
plt.show()

# Plot class distribution
plt.figure(figsize=(8, 5))
sns.countplot(data=data, x='Species')
plt.title('Class Distribution')
plt.show()

# Pairplot of features
sns.pairplot(data, hue='Species', palette='viridis')
plt.show()
