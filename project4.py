import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Create dataset
data = {
    "Loan_ID": range(1, 11),
    "ApplicantIncome": [4000,2500,6000,3000,7000,2000,5000,3500,8000,2800],
    "LoanAmount": [150,100,200,120,250,90,180,110,300,95],
    "Credit_History": [1,0,1,1,1,0,1,1,1,0],
    "Loan_Status": ["Approved","Rejected","Approved","Approved","Approved","Rejected","Approved","Approved","Approved","Rejected"]
}
df = pd.DataFrame(data)

X = df.drop(columns=['Loan_Status','Loan_ID'])
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Graph: Confusion Matrix
ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.show()
