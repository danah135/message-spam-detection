import pandas as pd

df = pd.read_csv("spam.csv", encoding="latin-1")

print(df[["v1","v2"]].head())

x=df["v2"]
y=df["v1"]
print("عدد الرسائل:",len(x))
print("التصنيفات المتاحة:",y.value_counts())

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_vec=vectorizer.fit_transform(x)
print("عدد الكلمات الفريدة:",X_vec.shape[1])
print("شكل المصفوفة:",X_vec.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)

print("عدد الرسائل في مجموعة التدريب:",X_train.shape[0])
print("عدد الرسائل في مجموعة الاختبار:",X_test.shape[0])
print()

model = MultinomialNB()
model.fit(X_train, y_train)
print("تم تدريب النموذج بنجاح!")

from sklearn.metrics import accuracy_score

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"دقة النموذج: {accuracy*100:1f}%")
print()


message=[

"congratulations! You've won a $1000 Walmart gift card. Go to http://bit.ly/123456 to claim now.",
"Hey, are we still on for dinner tonight?",
"URGENT! Your account has been compromised. Please reset your password immediately.",

]

print("تجربة تصنيف الرسائل الجديدة:")
for msg in message:
    msg_vec = vectorizer.transform([msg])
    prediction = model.predict(msg_vec)[0]
    print(f"الرسالة: {msg[:50]}...") 
    print(f"{ 'رسالة احتيالية' if prediction == 'spam' else 'رسالة غير احتيالية' }")

    import joblib
joblib.dump(model, "spam_model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")

