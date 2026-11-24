import pandas as pd
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# 1. قراءة البيانات (تأكد أن ملف Final_Cleaned_Disease_Data.csv موجود في نفس المجلد)
df = pd.read_csv('Final_Cleaned_Disease_Data.csv')
X = df.drop(columns=['diseases'])
y = df['diseases']

# 2. بناء وتدريب الخوارزميات الثلاث
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X, y)

nb_model = GaussianNB()
nb_model.fit(X, y)

knn_model = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
knn_model.fit(X.values, y) # استخدام X.values ضروري لتجنب تحذيرات مكتبة sklearn مع KNN

# 3. حفظ النماذج الثلاثة وقائمة الأعراض في ملفات مستقلة
joblib.dump(dt_model, 'dt_model.pkl')
joblib.dump(nb_model, 'nb_model.pkl')
joblib.dump(knn_model, 'knn_model.pkl')
joblib.dump(list(X.columns), 'symptoms_list.pkl')

print("✅ تم تدريب وحفظ الخوارزميات الثلاث (Decision Tree, Naive Bayes, KNN) بنجاح!")