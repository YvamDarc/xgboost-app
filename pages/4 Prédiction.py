import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import plotly.express as px
from io import BytesIO
from sklearn.preprocessing import MinMaxScaler

# Fonction de prétraitement des données
def preprocess_data(df, feature_cols):
    """
    Prépare les données pour la prédiction :
    - Convertit les colonnes datetime en timestamps.
    - Complète les colonnes manquantes avec des zéros.
    - Convertit les données en float.
    """
    df = df.copy()

    # Compléter les colonnes manquantes avec des zéros
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # Transformation temporelle si activée
    if st.session_state.get('temporal_features', False):
        def create_features(df, date_col):
            df = df.copy()
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            df['joursdelasemaine'] = df.index.dayofweek
            df['trimestre'] = df.index.quarter
            df['mois'] = df.index.month
            df['annee'] = df.index.year
            df['jourdelannee'] = df.index.dayofyear
            df['jourdumois'] = df.index.day
            df['semainedelannee'] = df.index.isocalendar().week.astype(float)
            df.reset_index(inplace=True)
            return df

        df = create_features(df, st.session_state.date_col)

    # Conversion des colonnes datetime en timestamps
    if st.session_state.date_col in df.columns:
        df[st.session_state.date_col] = pd.to_datetime(df[st.session_state.date_col])
        df[st.session_state.date_col] = df[st.session_state.date_col].apply(lambda x: x.timestamp())

    # Normalisation si activée
    if st.session_state.get('normalize', False):
        scaler = MinMaxScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Conversion des colonnes non numériques en float
    for col in df.columns:
        if df[col].dtype not in ['float64', 'int64', 'bool']:
            try:
                df[col] = df[col].astype(float)
            except ValueError:
                st.warning(f"Impossible de convertir la colonne {col} en float. Elle sera ignorée.")
                df.drop(columns=[col], inplace=True)

    return df

# Fonction pour ajouter les valeurs réelles et calculer les écarts
def add_real_values_and_calculate_differences(df_forecast):
    """
    Ajoute les valeurs réelles aux prédictions et calcule les écarts.
    """
    df_forecast = df_forecast.rename(columns={st.session_state.date_col: 'Date', 'prediction': 'Valeurs prédites'})

    # Ajouter une colonne pour les valeurs réelles si elle n'existe pas déjà
    if 'Valeur Réelle' not in df_forecast.columns:
        df_forecast['Valeur Réelle'] = None

    # Permettre à l'utilisateur de remplir les valeurs réelles
    df_forecast = st.data_editor(df_forecast)

    # Convertir la colonne 'Valeur Réelle' en numérique
    df_forecast['Valeur Réelle'] = pd.to_numeric(df_forecast['Valeur Réelle'], errors='coerce')

    # Calculer l'écart
    df_forecast['Écart'] = df_forecast['Valeur Réelle'] - df_forecast['Valeurs prédites']

    # Calculer les pourcentages d'écarts
    df_forecast['Pourcentage Écart'] = (df_forecast['Écart'] / df_forecast['Valeurs prédites']) * 100

    return df_forecast

# Titre de la page
st.title("Prédictions avec le Modèle XGBoost")

# Vérifier si le modèle et les données d'entraînement sont disponibles
if 'trained_model' in st.session_state and 'selected_features' in st.session_state:
    model = st.session_state['trained_model']
    feature_cols = st.session_state['selected_features']

    # Téléchargement des nouvelles données
    uploaded_file = st.file_uploader("Téléchargez un fichier CSV ou Excel pour les prédictions", type=["csv", "xlsx"])
    if uploaded_file:
        # Lire le fichier téléchargé
        file_extension = uploaded_file.name.split('.')[-1]
        if file_extension == 'csv':
            new_data = pd.read_csv(uploaded_file)
        elif file_extension == 'xlsx':
            new_data = pd.read_excel(uploaded_file)

        st.write("Nouvelles données chargées :")
        st.dataframe(new_data.head())

        # Prétraiter les nouvelles données
        X_new = preprocess_data(new_data, feature_cols)
        st.write("nouvelles features pour les prédictions :")
        st.dataframe(X_new.head())  

        # Vérifier si toutes les colonnes nécessaires sont présentes
        missing_cols = [col for col in feature_cols if col not in X_new.columns]
        if missing_cols:
            st.warning(f"Colonnes manquantes : {missing_cols}. Elles ont été complétées par des zéros.")

        # Prédictions
        try:
            predictions = model.predict(X_new[feature_cols])
            new_data['prediction'] = predictions

            st.write("Données avec prédictions :")
            st.dataframe(new_data)

            # Graphique des prédictions
            if st.session_state.date_col in new_data.columns:
                new_data[st.session_state.date_col] = pd.to_datetime(new_data[st.session_state.date_col], unit='s')
                fig = px.line(new_data, x=st.session_state.date_col, y='prediction', title="Prédictions du Modèle")
                st.plotly_chart(fig)

            # Cumul des prédictions
            total_predictions = new_data['prediction'].sum()
            st.write(f"Total des prédictions : {total_predictions}")

            # Analyse des écarts
            st.write("Exploitation des écarts entre les valeurs réelles et les prévisions :")
            df_with_real_values = add_real_values_and_calculate_differences(new_data[[st.session_state.date_col, 'prediction']])

            st.write("Tableau des écarts :")
            st.dataframe(df_with_real_values)

            # Graphique des écarts
            if 'Écart' in df_with_real_values.columns:
                fig_differences = px.line(df_with_real_values, x='Date', y='Écart', title='Écarts entre les valeurs réelles et les prévisions')
                st.plotly_chart(fig_differences)

            # Top pourcentage des écarts
            pourcentage = st.slider("Sélectionnez le pourcentage des plus gros écarts à afficher", min_value=1, max_value=50, value=15)
            top_n = max(1, round(len(df_with_real_values) * pourcentage / 100))

            plus_gros_positifs = df_with_real_values.nlargest(top_n, 'Écart')
            plus_gros_negatifs = df_with_real_values.nsmallest(top_n, 'Écart')

            st.subheader(f"Top {pourcentage}% des plus gros écarts positifs")
            st.dataframe(plus_gros_positifs)

            st.subheader(f"Top {pourcentage}% des plus gros écarts négatifs")
            st.dataframe(plus_gros_negatifs)

            # Téléchargement des résultats en Excel
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                new_data.to_excel(writer, sheet_name='Prédictions', index=False)
                df_with_real_values.to_excel(writer, sheet_name='Valeurs Réelles et Écarts', index=False)

            st.download_button(
                label="Télécharger les résultats en Excel",
                data=buffer.getvalue(),
                file_name="résultats_prédictions.xlsx",
                mime="application/vnd.ms-excel"
            )

        except Exception as e:
            st.error(f"Erreur lors des prédictions : {e}")
else:
    st.warning("Aucun modèle ou données d'entraînement disponibles. Veuillez revenir aux étapes précédentes pour configurer le modèle.")
