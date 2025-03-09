import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Veri setini yükleyin
data = pd.read_csv('heart_disease_uci.csv')

# 'id' ve 'dataset' sütunlarını çıkarın
data = data.drop(['id', 'dataset'], axis=1)

# Hedef değişkeni ve özellikleri ayırın
X = data.drop('num', axis=1)
y = data['num']

# Kategorik değişkenleri dönüştürün
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

# Verileri ölçeklendirin
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Verileri eğitim ve test setlerine ayırın
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Modeli oluşturun
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Modeli derleyin
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Modeli eğitin
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Modeli değerlendirin
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test doğruluğu: {accuracy * 100:.2f}%")

root = tk.Tk()
root.title("Kalp Hastalığı Tahmin Chatbot'u")

chat_log = scrolledtext.ScrolledText(root, width=80, height=20)
chat_log.pack(padx=10, pady=10)

user_input = tk.Entry(root, width=70)
user_input.pack(padx=10, pady=5)

def send_message():
    message = user_input.get()
    chat_log.config(state=tk.NORMAL)
    chat_log.insert(tk.END, "Siz: " + message + "\n")
    user_input.delete(0, tk.END)
    process_message(message)

send_button = tk.Button(root, text="Gönder", command=send_message)
send_button.pack(padx=10, pady=5)

questions = [
    "Yaşınızı girin: ",
    "Cinsiyetinizi girin (0: Kadın, 1: Erkek): ",
    "Göğüs ağrısı tipi (0-3 arası): ",
    "Dinlenme kan basıncınızı girin: ",
    "Kolesterol seviyenizi girin: ",
    "Açlık kan şekeri yüksek mi (0: Hayır, 1: Evet): ",
    "Dinlenme EKG durumu (0-2 arası): ",
    "Maksimum kalp atış hızınızı girin: ",
    "Egzersiz sırasında göğüs ağrısı var mı (0: Hayır, 1: Evet): ",
    "ST depresyonu değerini girin: ",
    "ST segment eğimi (0-2 arası): ",
    "Vasküler hastalık sayısı: ",
    "Thalassemia durumu (0-3 arası): "
]

answers = []
question_index = 0

def process_message(message):
    global question_index
    try:
        if question_index < len(questions):
            answers.append(int(message) if questions[question_index].startswith("Yaş") or questions[question_index].startswith("ST") else float(message))
            question_index += 1
            if question_index < len(questions):
                chat_log.insert(tk.END, "Chatbot: " + questions[question_index] + "\n")
            else:
                user_data_scaled = pd.DataFrame([answers], columns=X.columns)
                user_data_scaled = scaler.transform(user_data_scaled)
                prediction = model.predict(user_data_scaled)
                if prediction[0][0] >= 0.5:
                    response = "Kalp hastalığı riski var!"
                else:
                    response = "Kalp hastalığı riski yok."
                chat_log.insert(tk.END, "Chatbot: " + response + "\n")
        else:
            chat_log.insert(tk.END, "Chatbot: Tahmin yapıldı. Yeni bir tahmin için uygulamayı yeniden başlatın.\n")
    except Exception as e:
        chat_log.insert(tk.END, "Chatbot: Bir hata oluştu: " + str(e) + "\n")
    finally:
        chat_log.config(state=tk.DISABLED)
        chat_log.see(tk.END)

# İlk soruyu gönder
chat_log.insert(tk.END, "Chatbot: " + questions[question_index] + "\n")

root.mainloop()