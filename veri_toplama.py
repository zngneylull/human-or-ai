import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random

def arxiv_coklu_veri_cek(konu_listesi, hedef_toplam=3000):
    tum_veriler = []
    
    # Her konu için kaç tane çekilmesi gerektiğini hesapla
    konu_basi_hedef = hedef_toplam // len(konu_listesi)
    
    for konu in konu_listesi:
        print(f"\n--- '{konu}' konusu için veri toplanıyor (Hedef: {konu_basi_hedef}) ---")
        veriler_bu_konu = []
        sayfa_no = 0
        
        while len(veriler_bu_konu) < konu_basi_hedef:
            url = f"https://arxiv.org/search/?query={konu}&searchtype=all&source=header&start={sayfa_no}"
            
            try:
                response = requests.get(url)
                if response.status_code != 200:
                    break

                soup = BeautifulSoup(response.content, 'html.parser')
                makaleler = soup.find_all('li', class_='arxiv-result')
                
                if not makaleler:
                    break

                for makale in makaleler:
                    if len(veriler_bu_konu) >= konu_basi_hedef:
                        break
                    
                    try:
                        baslik = makale.find('p', class_='title').text.strip()
                        ozet_element = makale.find('span', class_='abstract-full')
                        if ozet_element:
                            ozet = ozet_element.text.replace('△ Less', '').strip()
                        else:
                            ozet = makale.find('span', class_='abstract-short').text.strip()

                        # Veriyi listeye ekle
                        veriler_bu_konu.append({
                            "Metin": ozet,
                            "Etiket": "Insan",
                            "Konu": konu,  # Hangi konuda olduğunu da kaydedelim, analizde işe yarar
                            "Kaynak": "Arxiv"
                        })
                        
                    except:
                        continue

                print(f"[{len(veriler_bu_konu)}/{konu_basi_hedef}] '{konu}' verisi çekildi.")
                sayfa_no += 50
                time.sleep(random.uniform(1.0, 2.0)) # Bekleme süresi

            except Exception as e:
                print(f"Hata: {e}")
                break
        
        # Bu konudaki verileri ana listeye ekle
        tum_veriler.extend(veriler_bu_konu)

    return pd.DataFrame(tum_veriler)

# --- ÇALIŞTIRMA KISMI ---
if __name__ == "__main__":
    # 5 Farklı ve zıt konu seçtik ki model ezber yapamasın
    konular = [
        "Artificial Intelligence",  # Bilgisayar
        "Quantum Physics",          # Fizik
        "Cell Biology",             # Biyoloji
        "Economics",                # Ekonomi
        "History of Art"            # Sosyal Bilimler
    ]
    
    # Toplam 3000 veri (Her konudan 600 tane)
    df = arxiv_coklu_veri_cek(konular, hedef_toplam=3000)
    
    # Excel'e kaydet
    df.to_excel("insan_veri_seti_karisik.xlsx", index=False)
    print(f"\nToplam {len(df)} veri başarıyla kaydedildi!")