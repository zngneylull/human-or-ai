import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def modelleri_yaristir():
    print("--- 3 FARKLI ALGORİTMA İLE MODEL EĞİTİMİ ---")

    # Veriyi Yükleme
    print("Veriler yükleniyor...")
    try:
        df = pd.read_csv("final_proje_verisi.csv")
        df = df.dropna()
    except FileNotFoundError:
        print("HATA: 'final_proje_verisi.csv' yok! Önce temizleme kodunu çalıştır.")
        return

    X = df['Metin']
    y = df['Etiket']

    # Eğitim/Test Ayrımı
    print("Veri bölünüyor (%80 Eğitim - %20 Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vektörleştirme (TF-IDF)
    print("Metinler sayısallaştırılıyor (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Modelleri Tanımlama
    print("Modeller tanımlanıyor...")
    modeller = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    sonuclar = {}
    en_iyi_model = None
    en_yuksek_basari = 0
    en_iyi_model_ismi = ""

    print("\nModeller eğitiliyor ve test ediliyor...")
    
    for isim, model in modeller.items():
        print(f"\n>> {isim} modeli eğitiliyor...")
        
        # Eğitme
        model.fit(X_train_vec, y_train)
        
        # Test Etme
        print("Model test ediliyor...")
        tahminler = model.predict(X_test_vec)
        basari = accuracy_score(y_test, tahminler)
        
        # En iyi modeli saklama
        if basari > en_yuksek_basari:
            en_yuksek_basari = basari
            en_iyi_model = model
            en_iyi_model_ismi = isim

    # Sonucu Raporlama
        sonuclar[isim] = basari
        print(f"\n🏆 {isim} Başarı Oranı (Accuracy): %{basari * 100:.2f}")
    
    print("\n--- Detaylı Sınıflandırma Raporu ---")
    print(classification_report(y_test, tahminler, target_names=['İnsan', 'AI']))

    # --- SONUÇLARI GÖRSELLEŞTİRME - Karşılaştırma Grafiği ---
    print("\n📊 Karşılaştırma Grafiği Çiziliyor...")
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(sonuclar.keys()), y=list(sonuclar.values()), palette="viridis")
    plt.title('Algoritma Karşılaştırması')
    plt.ylim(0.8, 1.0) # Grafiği 0.8'den başlat ki farklar net görünsün
    plt.ylabel('Başarı Oranı')
    plt.savefig('algoritma_karsilastirma.png')
    print("📊 Karşılaştırma Grafiği kaydedildi: 'algoritma_karsilastirma.png'")

    # En İyi Modelin Detaylı Analizi
    # Confusion Matrix (Karmaşıklık Matrisi) Görselleştirme
    print(f"\n--- {en_iyi_model_ismi} İçin Detaylı Analiz ---")
    en_iyi_model_tahmin = en_iyi_model.predict(X_test_vec)
    
    print(classification_report(y_test, en_iyi_model_tahmin, target_names=['İnsan (0)', 'AI (1)']))

    # Confusion Matrix Çizimi
    cm = confusion_matrix(y_test, en_iyi_model_tahmin)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tahmin: İnsan', 'Tahmin: AI'], yticklabels=['Gerçek: İnsan', 'Gerçek: AI'])
    plt.title('Confusion Matrix (Hata Tablosu)')
    plt.ylabel('Gerçek Durum')
    plt.xlabel('Modelin Tahmini')
    plt.savefig('confusion_matrix.png')
    print("📊 Karmnaşıklık Matrisi kaydedildi: 'confusion_matrix.png'")

    # --- EN İYİ MODELİ KAYDETME ---
    print(f"\n🌟 EN İYİ MODEL: {en_iyi_model_ismi} (%{en_yuksek_basari*100:.2f})")
    print("En iyi model ve vektörleyici kaydediliyor...")
    
    joblib.dump(en_iyi_model, 'best_model.pkl')
    joblib.dump(vectorizer, 'vectorizer.pkl')
    
    print("💾 Dosyalar hazır: 'best_model.pkl' ve 'vectorizer.pkl'")   
    print("User Story-3 Tamamlandı!")

if __name__ == "__main__":
    modelleri_yaristir()