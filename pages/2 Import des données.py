import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Initialisation de l'état de session pour les données
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_filtered' not in st.session_state:
    st.session_state.df_filtered = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'temporal_features_added' not in st.session_state:
    st.session_state.temporal_features_added = False
if 'data_normalized' not in st.session_state:
    st.session_state.data_normalized = False

# Interface de l'application Streamlit
st.title("Prévision avec XGBoost")

# Téléchargement des données par l'utilisateur
uploaded_file = st.file_uploader("Téléchargez votre fichier CSV ou Excel", type=["csv", "xlsx"])
if uploaded_file:
    # Déterminer le type de fichier et le lire
    file_extension = uploaded_file.name.split('.')[-1]

    if file_extension == 'csv':
        df = pd.read_csv(uploaded_file)
    elif file_extension == 'xlsx':
        df = pd.read_excel(uploaded_file)

    st.session_state.df = df
    st.write(df.head())

# Vérifier si les données sont chargées
if st.session_state.df is not None:
    df = st.session_state.df

    # Sélection des colonnes
    date_col = st.selectbox("Sélectionnez la colonne de date", df.columns)
    target_col = st.selectbox("Sélectionnez la colonne cible", df.columns)

    # Vérification et mise à jour de selected_features
    if 'selected_features' not in st.session_state or not st.session_state.selected_features:
        st.session_state.selected_features = []

    # Bouton pour sélectionner toutes les colonnes explicatives
    if st.button("Sélectionner toutes les colonnes explicatives"):
        st.session_state.selected_features = [col for col in df.columns if col not in [date_col, target_col]]

    # Interface pour sélectionner/désélectionner les colonnes explicatives
    st.write("Sélectionnez les colonnes des variables explicatives :")
    columns_per_row = 4
    cols = st.columns(columns_per_row)

    updated_selected_features = []

    for idx, column in enumerate(df.columns):
        if column not in [date_col, target_col]:
            if idx % columns_per_row == 0 and idx != 0:
                cols = st.columns(columns_per_row)
            selected = cols[idx % columns_per_row].checkbox(
                column, value=(column in st.session_state.selected_features), key=f"checkbox_{column}"
            )
            if selected:
                updated_selected_features.append(column)

    st.session_state.selected_features = updated_selected_features

    # Afficher les colonnes sélectionnées ou un message si aucune n'est sélectionnée
    if st.session_state.selected_features:
        st.write(f"Colonnes explicatives sélectionnées : **{st.session_state.selected_features}**")
    else:
        st.write("Aucune colonne explicative sélectionnée.")

    # Inclure uniquement la colonne de date, les features, et la cible dans df
    feature_cols = st.session_state.selected_features
    if date_col and target_col:
        if feature_cols:
            df_filtered = df[[date_col] + feature_cols + [target_col]]
        else:
            df_filtered = df[[date_col, target_col]]  # Inclure seulement la date et la cible si aucune feature n'est sélectionnée

        st.write("Données sélectionnées (Date, Features, Cible) :")
        st.dataframe(df_filtered.head())
    else:
        st.warning("Veuillez sélectionner les colonnes nécessaires (date, features, et cible).")

    # Tri des données par la colonne de date
    if date_col in df_filtered.columns:
        df_filtered = df_filtered.sort_values(by=date_col)

    # Filtrage des dates pour l'apprentissage
    if date_col in df_filtered.columns:
        st.write("Filtrage des dates pour l'apprentissage")

        # Convertir la colonne de date si nécessaire
        df_filtered[date_col] = pd.to_datetime(df_filtered[date_col])

        # Sélection de la période pour filtrer
        start_date = st.date_input("Date de début pour l'entraînement", df_filtered[date_col].min().date())
        end_date = st.date_input("Date de fin pour l'entraînement", df_filtered[date_col].max().date())

        if start_date and end_date:
            # Vérifier la validité des dates
            if start_date > end_date:
                st.error("La date de début ne peut pas être après la date de fin.")
            else:
                # Filtrer les données dans la plage de dates
                mask = (df_filtered[date_col] >= pd.Timestamp(start_date)) & (df_filtered[date_col] <= pd.Timestamp(end_date))
                df_filtered = df_filtered[mask]

                if df_filtered.empty:
                    st.error("Aucun échantillon dans le DataFrame filtré. Veuillez ajuster les dates.")
                else:
                    st.write("Jeu de données filtré :")
                    st.dataframe(df_filtered.head())

    # Aperçu global des données filtrées
    if 'df_filtered' in locals() and not df_filtered.empty:
        st.write("Aperçu des données prêtes pour l'entraînement :")
        st.dataframe(df_filtered.head())

        # Bouton pour créer des caractéristiques temporelles
        temporal_button = st.checkbox("Ajouter des caractéristiques temporelles", value=False)
        normalize_button = st.checkbox("Normaliser les données", value=False)

        if temporal_button:
            try:
                def create_features(df, date_col):
                    """
                    Crée des caractéristiques temporelles à partir de la colonne de date.
                    """
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

                df_filtered = create_features(df_filtered, date_col)

                # Ajouter les nouvelles colonnes générées à selected_features
                new_features = [col for col in df_filtered.columns if col not in [date_col, target_col] + feature_cols]
                st.session_state.selected_features.extend(new_features)
                st.write(f"Colonnes explicatives après ajout des caractéristiques temporelles : **{st.session_state.selected_features}**")

                st.session_state.temporal_features_added = True
                st.dataframe(df_filtered.head())
            except Exception as e:
                st.error(f"Erreur lors de la création des caractéristiques temporelles : {e}")

        if normalize_button:
            try:
                scaler = MinMaxScaler()
                df_filtered[feature_cols] = scaler.fit_transform(df_filtered[feature_cols])
                st.write("Données normalisées :")
                st.session_state.data_normalized = True
                st.dataframe(df_filtered.head())
            except Exception as e:
                st.error(f"Erreur lors de la normalisation des données : {e}")

        # Enregistrer les modifications dans la session
        st.session_state.df_filtered = df_filtered

        # Sauvegarde des colonnes cibles et explicatives
        st.session_state.target_col = target_col
        st.session_state.date_col = date_col

        st.write(f"Colonne cible sélectionnée : **{target_col}**")
        st.write(f"Colonnes explicatives sélectionnées : **{st.session_state.selected_features}**")

        # Visualisation des données finales
        st.write("Visualisation des données finales :")
        st.dataframe(df_filtered)

        # Bouton pour enregistrer les données en fichier Excel
        save_button = st.button("Enregistrer les données finalisées en fichier Excel")
        if save_button:
            output_file = "donnees_finalisees.xlsx"
            try:
                df_filtered.to_excel(output_file, index=False)
                st.success(f"Fichier enregistré avec succès : {output_file}")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement du fichier : {e}")
