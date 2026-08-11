#exp3
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

data = pd.DataFrame([
    ['Sunny', 'Hot', 'High', 'Weak', 'No'],
    ['Sunny', 'Hot', 'High', 'Strong', 'No'],
    ['Overcast', 'Hot', 'High', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'High', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Cool', 'Normal', 'Strong', 'No'],
    ['Overcast', 'Cool', 'Normal', 'Strong', 'Yes'],
    ['Sunny', 'Mild', 'High', 'Weak', 'No'],
    ['Sunny', 'Cool', 'Normal', 'Weak', 'Yes'],
    ['Rain', 'Mild', 'Normal', 'Weak', 'Yes']
], columns=['Outlook', 'Temperature', 'Humidity', 'Wind', 'PlayTennis'])

le = LabelEncoder()
encoded_data = data.apply(le.fit_transform)

X = encoded_data.iloc[:, :-1]
y = encoded_data.iloc[:, -1]

clf = DecisionTreeClassifier(criterion='entropy')
clf.fit(X, y)

print("Decision Tree Rules:")
print(export_text(clf, feature_names=list(X.columns)))

sample = [[2, 1, 0, 1]]
pred = clf.predict(sample)
print("Prediction for new sample", sample, ":", pred[0])
