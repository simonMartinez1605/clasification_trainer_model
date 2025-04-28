import os
import re
import time
import nltk
import shutil
import joblib
import pytesseract
from nltk.corpus import stopwords
from pdf2image import convert_from_path
from sklearn.metrics import classification_report
from concurrent.futures import ProcessPoolExecutor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score


nltk.download('stopwords')

# Configura Tesseract y Poppler
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Cambia si es necesario
poppler_path = r'C:\Users\simon\Downloads\Release-24.08.0-0\poppler-24.08.0\Library\bin'  # Cambia si es necesario

# Directorios
source_dir = r"C:\Users\simon\OneDrive\Documents\Simon\Training\Classification\raw_documents"
destination_dir = r"C:\Users\simon\OneDrive\Documents\Simon\Training\Classification\sorted_documents"
model_path = 'modelo_clasificador.pkl'

# Cargar etiquetas reales si existen (esto es para un futuro dataset real)
# En este ejemplo siguen siendo etiquetas dummy
etiquetas_posibles = ['Appointment', 'Approval', 'Defensive']

# Funcion para limpiar texto de manera mas inteligente
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^A-Za-z0-9 .,;:/\n-]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

# Extraer caracteristicas adicionales
def extract_features(text):
    return {
        'text': text,
        'notice_count': text.count('notice'),
        'approval_count': text.count('approval'),
        'hearing_count': text.count('hearing'),
        'date_count': len(re.findall(r'\d{2}/\d{2}/\d{4}', text))
    }

# Funcion para extraer texto de un PDF usando OCR
def extract_text_from_pdf(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=200, poppler_path=poppler_path)
        text = ""
        for img in images:
            text += pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error extrayendo {pdf_path}: {e}")
        return ""

# Extraer texto de todos los PDFs
def extract_texts_from_pdfs(directory):
    pdf_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.pdf')]
    with ProcessPoolExecutor(max_workers=4) as executor:
        texts = list(executor.map(extract_text_from_pdf, pdf_files))
    return texts, pdf_files

# Entrenar modelo
def train_model(texts, labels):
    features = [extract_features(clean_text(text)) for text in texts]
    corpus = [f['text'] for f in features]
    vectorizer = TfidfVectorizer(max_features=2000)
    X_text = vectorizer.fit_transform(corpus)

    # Combinar features adicionales (por simplicidad, aquí solo texto)
    X = X_text
    y = labels

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=72)

    clf = GradientBoostingClassifier()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    scores = cross_val_score(clf, X, y, cv=2)
    print(f"Cross-validation scores: {scores}")
    print(f"Mean CV score: {scores.mean():.2f}")

    joblib.dump((vectorizer, clf), model_path)

# Clasificar nuevo PDF
def classify_pdf(pdf_path, vectorizer, clf):
    text = extract_text_from_pdf(pdf_path)
    cleaned_text = clean_text(text)
    features = extract_features(cleaned_text)
    X_text = vectorizer.transform([features['text']])
    prediction = clf.predict(X_text)
    return prediction[0]

# Clasificar todos los PDFs
if __name__ == "__main__":
    start_time = time.time()

    # Paso 1: Extraer textos de PDFs
    texts, pdf_files = extract_texts_from_pdfs(source_dir)

    # Paso 2: Crear etiquetas dummy (sustituir luego por reales)
    labels = (etiquetas_posibles * (len(texts) // len(etiquetas_posibles) + 1))[:len(texts)]

    # Paso 3: Entrenar y guardar el modelo
    train_model(texts, labels)

    # Paso 4: Cargar modelo entrenado
    vectorizer, clf = joblib.load(model_path)

    # Paso 5: Clasificar y mover PDFs
    for pdf_path in pdf_files:
        try:
            prediction = classify_pdf(pdf_path, vectorizer, clf)
            print(f"Clasificado: {os.path.basename(pdf_path)} -> {prediction}")

            destino_folder = os.path.join(destination_dir, prediction)
            os.makedirs(destino_folder, exist_ok=True)
            shutil.move(pdf_path, os.path.join(destino_folder, os.path.basename(pdf_path)))
        except Exception as e:
            print(f"Error clasificando {pdf_path}: {e}")

    print(f"Proceso terminado en {time.time() - start_time:.2f} segundos.")