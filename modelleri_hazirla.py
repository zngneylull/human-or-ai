import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def modelleri_kaydet():
    print("--- ÇOKLU MODEL DESTEĞİ İÇİN HAZIRLIK ---")
    
    # 1. Veriyi Oku
    try:
        df = pd.read_csv("final_proje_verisi.csv").dropna()
    except:
        print("Hata: final_proje_verisi.csv yok!")
        return

    X = df['Metin']
    y = df['Etiket']

    # 2. Vektörleştiriciyi Eğit ve Kaydet
    print("1. Vektörleştirici hazırlanıyor...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X) # Tüm veriyle eğitiyoruz (Final ürünü olduğu için)
    joblib.dump(vectorizer, 'vectorizer.pkl')

    # 3. Üç Modeli de Ayrı Ayrı Eğit ve Kaydet
    print("2. Modeller kaydediliyor...")

    # Naive Bayes
    nb = MultinomialNB()
    nb.fit(X_vec, y)
    joblib.dump(nb, 'model_naive_bayes.pkl')
    print("   ✅ Naive Bayes kaydedildi.")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_vec, y)
    joblib.dump(lr, 'model_logistic.pkl')
    print("   ✅ Logistic Regression kaydedildi.")

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_vec, y)
    joblib.dump(rf, 'model_random_forest.pkl')
    print("   ✅ Random Forest kaydedildi.")

    print("\n🎉 HAZIRLIK TAMAM! Artık 3 modelimiz de dosya olarak elimizde.")

if __name__ == "__main__":
    modelleri_kaydet()