import tkinter as tk
from tkinter import scrolledtext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Veri setini yükleyin
data = pd.read_csv('heart.csv')

# Hedef değişkeni ve özellikleri ayırın
X = data.drop('HeartDisease', axis=1)
y = data['HeartDisease']

# Kategorik değişkenleri One-Hot Encoding ile dönüştürün
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
encoder = OneHotEncoder(sparse_output=False, drop='first')
X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]))
X_encoded.columns = encoder.get_feature_names_out(categorical_cols)
X = pd.concat([X.drop(categorical_cols, axis=1), X_encoded], axis=1)

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
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Modeli değerlendirin
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test doğruluğu: {accuracy * 100:.2f}%")

# --- Tkinter arayüzü ---

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
    "Cinsiyetinizi girin (K/E): ",
    "Göğüs ağrısı tipinizi girin (ATA/NAP/ASY/TA): ",
    "Dinlenme kan basıncınızı girin: ",
    "Kolesterol seviyenizi girin: ",
    "Açlık kan şekeriniz yüksek mi (0/1): ",
    "Dinlenme EKG durumunuzu girin (Normal/ST/LVH): ",
    "Maksimum kalp atış hızınızı girin: ",
    "Egzersiz sırasında göğüs ağrınız var mı (V/Y): ",
    "Oldpeak değerinizi girin: ",
    "ST segment eğiminizi girin (Up/Flat/Down): "
]

answers = []
question_index = 0

def process_message(message):
    global question_index
    try:
        if question_index < len(questions):
            if question_index == 0 or question_index == 3 or question_index == 4 or question_index == 7 or question_index == 9:
                try:
                    answers.append(float(message))
                except ValueError:
                    chat_log.insert(tk.END, "Chatbot: Geçersiz giriş. Lütfen sayısal bir değer girin.\n")
                    return
            else:
                answers.append(message)
            question_index += 1
            if question_index < len(questions):
                chat_log.insert(tk.END, "Chatbot: " + questions[question_index] + "\n")
            else:
                user_data = pd.DataFrame([answers], columns=X.columns[:11])
                user_data_encoded = pd.DataFrame(encoder.transform(user_data[categorical_cols]))
                user_data_encoded.columns = encoder.get_feature_names_out(categorical_cols)
                user_data = pd.concat([user_data.drop(categorical_cols, axis=1), user_data_encoded], axis=1)
                user_data_scaled = scaler.transform(user_data)
                prediction = model.predict(user_data_scaled, verbose=0)
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