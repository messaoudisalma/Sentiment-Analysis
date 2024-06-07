import streamlit as st
import joblib
import numpy as np

# Charger le modèle et le vectorizer
nb_model = joblib.load('nb_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
best_threshold = joblib.load('best_threshold.pkl')

# CSS de base avec un fond gris et une marge à gauche
st.markdown("""
    <style>
    .main {
        background-color: #f0f0f0;
        padding: 20px;
        transition: background-image 0.5s ease-in-out;
    }
    .title {
        font-family: 'Helvetica Neue', sans-serif;
        color: #4CAF50;
        text-align: center;
    }
    .subtitle {
        font-family: 'Helvetica Neue', sans-serif;
        color: #777;
        text-align: center;
        font-size: 20px;
    }
    .description {
        font-family: 'Helvetica Neue', sans-serif;
        color: #444;
        text-align: center;
    }
    .comment-area {
        font-family: 'Helvetica Neue', sans-serif;
        color: #333;
        padding: 10px;
        border: 1px solid #ddd;
        border-radius: 5px;
    }
    .prediction-result {
        font-family: 'Helvetica Neue', sans-serif;
        font-size: 20px;
        text-align: center;
        margin-top: 20px;
    }
    .footer {
        font-family: 'Helvetica Neue', sans-serif;
        color: #aaa;
        text-align: center;
        margin-top: 50px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextArea textarea {
        height: 150px;
    }
    .card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    .background {
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }
    .sidebar {
        font-size: 24px;
        padding: 20px;
        background-color: #ffffff;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# Titre de l'application avec des emojis
st.markdown('<h1 class="title">📝 Prédiction des Sentiments des Commentaires 📊</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Analyse de Sentiment en Temps Réel 😊😢</h3>', unsafe_allow_html=True)

# Description de l'application avec un emoji
st.markdown(
    """
    <p class="description">
    Cette application utilise un modèle de machine learning pour prédire le sentiment d'un commentaire donné.
    Entrez un commentaire ci-dessous et appuyez sur le bouton <b>Prédire</b> pour voir la prédiction.
    </p>
    """, unsafe_allow_html=True
)

# Ajout d'un espace réservé pour le sentiment prédit dans la marge à gauche
prediction_placeholder = st.sidebar.empty()

# Saisie utilisateur avec un meilleur label

user_input = st.text_area('✍️ Entrez votre commentaire ici:', '', key='comment', help='Tapez le commentaire pour lequel vous souhaitez prédire le sentiment')

# Bouton de prédiction avec un emoji
if st.button('Prédire 🔍'):
    if user_input:
        # Vectoriser le commentaire utilisateur
        user_input_vectorized = vectorizer.transform([user_input])
        
        # Prédiction des probabilités
        prediction_proba = nb_model.predict_proba(user_input_vectorized)
        
        # Classification basée sur le seuil ajusté
        prediction_adjusted = (prediction_proba[:, 1] >= best_threshold).astype(int)
        
        # Afficher le résultat avec une couleur différente selon le sentiment
        unique_classes = nb_model.classes_
        prediction_label = unique_classes[prediction_adjusted[0]]
        
        if prediction_label == 'positif':
            st.markdown(f'<p class="prediction-result" style="color: green;"> Le modèle prédit que le commentaire est <b>{prediction_label}</b>.</p>', unsafe_allow_html=True)
            st.markdown(
                """
                <script>
                document.getElementById("bg").style.backgroundImage = "";
                document.getElementById("bg").style.backgroundColor = "#d4edda";
                </script>
                """, unsafe_allow_html=True
            )
        else:
            st.markdown(f'<p class="prediction-result" style="color: black;"> Le modèle prédit que le commentaire est <b>{prediction_label}</b>.</p>', unsafe_allow_html=True)
            st.markdown(
                """
                <script>
                document.getElementById("bg").style.backgroundImage = "";
                document.getElementById("bg").style.backgroundColor = "#f8d7da";
                </script>
                """, unsafe_allow_html=True
            )
        
        # Mettre à jour le sentiment prédit dans la marge à gauche
        prediction_placeholder.markdown(f'<div class="sidebar" style="color:black;">Sentiment prédit: {prediction_label} 😊</div>', unsafe_allow_html=True)
    else:
        st.warning('Veuillez entrer un commentaire pour faire une prédiction.')
st.markdown('</div>', unsafe_allow_html=True)

# Ajouter un pied de page avec un emoji
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    """
    <p class="footer">
    Application développée avec Streamlit. 🌐
    </p>
    """, unsafe_allow_html=True
)
