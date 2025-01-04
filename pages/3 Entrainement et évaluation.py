import streamlit as st
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
import plotly.express as px
from xgboost import plot_importance
import matplotlib.pyplot as plt

# Fonction de prétraitement
def preprocess_data(df):
    """
    Convertit les colonnes datetime en timestamps et transforme les colonnes non numériques en float.
    """
    df = df.copy()

    # Conversion des colonnes datetime en timestamps
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].apply(lambda x: x.timestamp())

    # Conversion des colonnes non numériques en float
    for col in df.select_dtypes(exclude=['number']).columns:
        try:
            df[col] = df[col].astype(float)
        except ValueError:
            st.warning(f"Impossible de convertir la colonne {col} en float. Elle sera ignorée.")
            df.drop(columns=[col], inplace=True)

    return df

# Fonction de cross-validation classique
def perform_cross_validation(X, y, params, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    results = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        smape = 100 * np.mean(2 * np.abs(predictions - y_test) / (np.abs(predictions) + np.abs(y_test)))
        results.append({"RMSE": rmse, "MAE": mae, "SMAPE": smape})

    df_results = pd.DataFrame(results)
    df_results['horizon'] = range(1, len(df_results) + 1)
    return df_results

# Fonction de cross-validation cumulée
def cumulative_cross_validation(X, y, params, initial, period, horizon):
    """
    Effectue une cross-validation cumulée pour une série temporelle avec XGBoost.
    """
    results = []

    start_idx = 0
    while start_idx + initial + horizon <= len(X):
        train_idx = slice(start_idx, start_idx + initial)
        test_idx = slice(start_idx + initial, start_idx + initial + horizon)

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Calcul des métriques cumulées
        cumsum_y = y_test.cumsum()
        cumsum_yhat = pd.Series(predictions).cumsum()
        cumsum_error = cumsum_y - cumsum_yhat

        results.append({
            'Fin de période': test_idx.stop,
            'Cumulative RMSE': np.sqrt(mean_squared_error(y_test, predictions)),
            'Cumulative MAE': mean_absolute_error(y_test, predictions),
            'Cumulative SMAPE': 100 * np.mean(2 * np.abs(predictions - y_test) / (np.abs(predictions) + np.abs(y_test)))
        })

        start_idx += period

    return pd.DataFrame(results)

# Fonction de Grid Search
def grid_search_xgboost(X, y, param_grid, metric='rmse', n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    all_params = list(ParameterGrid(param_grid))
    best_params = None
    best_metric_value = float('inf')
    results = []

    for params in all_params:
        metrics = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            if metric == 'rmse':
                value = np.sqrt(mean_squared_error(y_test, predictions))
            elif metric == 'mae':
                value = mean_absolute_error(y_test, predictions)
            elif metric == 'smape':
                value = 100 * np.mean(2 * np.abs(predictions - y_test) / (np.abs(predictions) + np.abs(y_test)))
            else:
                raise ValueError("Métrique non supportée.")

            metrics.append(value)

        mean_metric = np.mean(metrics)
        results.append({"params": params, f"mean_{metric}": mean_metric})

        if mean_metric < best_metric_value:
            best_metric_value = mean_metric
            best_params = params

    return best_params, pd.DataFrame(results)

# Fonction pour calculer les métriques de validation
def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true)))
    return {"RMSE": rmse, "MAE": mae, "SMAPE": smape}

# Fonction pour générer la validation curve
def validation_curve(X_train, y_train, X_test, y_test, param_name, param_values):
    """
    Génère une courbe de validation pour un paramètre donné.
    """
    train_scores = []
    test_scores = []

    for value in param_values:
        params = {
            "n_estimators": value,
            "max_depth": 3,
            "learning_rate": 0.1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "gamma": 0.0
        }
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        # Prédictions sur train et test
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Calcul des métriques
        train_rmse, _, _ = calculate_metrics(y_train, train_pred)
        test_rmse, _, _ = calculate_metrics(y_test, test_pred)

        train_scores.append(train_rmse)
        test_scores.append(test_rmse)

    return param_values, train_scores, test_scores

# Titre de la page
st.title("Entraînement du Modèle XGBoost")

# Vérifier si les données sont disponibles dans st.session_state
if 'df_filtered' in st.session_state and 'selected_features' in st.session_state:
    df_filtered = st.session_state.df_filtered
    feature_cols = st.session_state.selected_features
    target_col = st.session_state.target_col

    st.write("Données récupérées pour l'entraînement :")
    st.dataframe(df_filtered.head())
    st.write("features récupérées pour l'entraînement :")
    st.dataframe(df_filtered[feature_cols].head())
    st.write("target récupérées pour l'entraînement :")
    st.dataframe(df_filtered[target_col].head())   

    if target_col and feature_cols:
        # Prétraiter les données
        df_filtered = preprocess_data(df_filtered)

        X = df_filtered[feature_cols]
        y = df_filtered[target_col]

        # Entraînement initial
        st.write("Entraînement initial du modèle XGBoost")
        params = {
            "n_estimators": st.number_input("Nombre d'arbres", value=100),
            "max_depth": st.number_input("Profondeur maximale des arbres", value=3),
            "learning_rate": st.number_input("Taux d'apprentissage", value=0.1),
            "subsample": st.number_input("Ratio d'échantillonnage", value=1.0),
            "colsample_bytree": st.number_input("Échantillonnage des colonnes par arbre", value=1.0),
            "gamma": st.number_input("Gamma", value=0.0)
        }

        # Sélection de la taille du test set
        test_size = st.number_input("Taille de l'ensemble de test (jours)", value=30, min_value=1, max_value=len(X))

        if st.button("Entraîner le Modèle XGBoost"):
            try:
                # Séparation des données en train et test
                X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
                y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]

                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)

                # Prédictions sur l'ensemble de test
                y_pred = model.predict(X_test)

                # Sauvegarder le modèle et les données d'entraînement
                st.session_state['trained_model'] = model
                st.session_state['X_train'] = X_train
                st.session_state['y_train'] = y_train

                st.success("Modèle entraîné avec succès.")
                
                # Graphique des features importantes
                st.write("Graphique des Features Importantes :")
                fig, ax = plt.subplots(figsize=(10, 8))
                plot_importance(model, ax=ax, height=0.9)
                st.pyplot(fig)

                # Affichage des prédictions sur l'ensemble de test
                st.write("Résultats sur l'ensemble de test :")
                test_results = pd.DataFrame({
                    st.session_state.date_col: df_filtered[st.session_state.date_col].iloc[-test_size:],
                    "Valeurs Réelles": y_test,
                    "Prédictions": y_pred
                })
                st.dataframe(test_results)

                # Graphique des prédictions
                fig_test = px.line(test_results, x=st.session_state.date_col, y=["Valeurs Réelles", "Prédictions"],
                                   title="Prédictions vs Valeurs Réelles sur l'ensemble de test")
                st.plotly_chart(fig_test)

                # Calcul des métriques sur l'ensemble de test
                metrics = calculate_metrics(y_test, y_pred)
                st.write("Métriques sur l'ensemble de test :")
                st.write(f"RMSE : {metrics['RMSE']}")
                st.write(f"MAE : {metrics['MAE']}")
                st.write(f"SMAPE : {metrics['SMAPE']}")

            except Exception as e:
                st.error(f"Erreur lors de l'entraînement : {e}")

        # Sélection du type de validation croisée
        cross_val_type = st.selectbox("Type de validation croisée :", ["Classique", "Cumulée", "GridSearch"])

        if cross_val_type == "Classique":
            # Validation croisée classique
            with st.form(key='cv_form'):
                st.write("Définir les périodes pour la validation croisée")
                initial = st.number_input("Période initiale (jours)", value=365, min_value=1)
                period = st.number_input("Période de validation (jours)", value=30, min_value=1)
                horizon = st.number_input("Horizon de prédiction (jours)", value=30, min_value=1)

                if st.form_submit_button("Lancer la validation croisée"):
                    try:
                        cv_results = perform_cross_validation(X, y, params)
                        st.session_state.cv_results = cv_results
                        st.success("Validation croisée effectuée avec succès.")
                    except Exception as e:
                        st.error(f"Erreur lors de la validation croisée : {e}")

            if 'cv_results' in st.session_state and st.session_state.cv_results is not None:
                cv_results = st.session_state.cv_results
                st.write("Résultats de la Validation Croisée :")
                st.dataframe(cv_results)

                csv2 = cv_results.to_csv(index=False)
                st.download_button(
                    label="Télécharger le diagnostic",
                    data=csv2,
                    file_name='diag.csv',
                    mime='text/csv'
                )

                if "selected_metric" not in st.session_state:
                    st.session_state["selected_metric"] = "SMAPE"

                st.session_state["selected_metric"] = st.selectbox(
                    "Sélectionnez la colonne à afficher dans le graphique :",
                    cv_results.columns,
                    index=cv_results.columns.get_loc(st.session_state["selected_metric"])
                    if st.session_state["selected_metric"] in cv_results.columns
                    else 0,
                )

                if st.session_state["selected_metric"]:
                    fig_performance = px.line(
                        cv_results,
                        x='horizon',
                        y=st.session_state["selected_metric"],
                        title=f'{st.session_state["selected_metric"]} sur la période de validation'
                    )
                    st.plotly_chart(fig_performance)
                    
        elif cross_val_type == "Cumulée":
            # Validation croisée cumulée
            with st.form(key='cumulative_cv_form'):
                st.write("Définir les périodes pour la validation croisée cumulée")
                initial = st.number_input("Période initiale (jours)", value=365, min_value=1)
                period = st.number_input("Période de validation (jours)", value=30, min_value=1)
                horizon = st.number_input("Horizon de prédiction (jours)", value=30, min_value=1)

                if st.form_submit_button("Lancer la validation croisée cumulée"):
                    try:
                        cumulative_results = cumulative_cross_validation(X, y, params, initial, period, horizon)
                        st.session_state['cumulative_cv_results'] = cumulative_results
                        st.success("Validation croisée cumulée effectuée avec succès.")
                    except Exception as e:
                        st.error(f"Erreur lors de la validation croisée cumulée : {e}")

            if 'cumulative_cv_results' in st.session_state and st.session_state.cumulative_cv_results is not None:
                cumulative_results = st.session_state.cumulative_cv_results
                st.write("Résultats de la Validation Croisée Cumulée :")
                st.dataframe(cumulative_results)

                csv_cumulative = cumulative_results.to_csv(index=False)
                st.download_button(
                    label="Télécharger les résultats cumulés",
                    data=csv_cumulative,
                    file_name='cumulative_metrics.csv',
                    mime='text/csv'
                )

                column_to_plot = st.selectbox(
                    "Sélectionnez la colonne à afficher dans le graphique :",
                    cumulative_results.columns,
                    index=cumulative_results.columns.get_loc('Cumulative MAE')
                    if 'Cumulative MAE' in cumulative_results.columns else 0
                )

                fig_cumulative_performance = px.line(
                    cumulative_results,
                    x='Fin de période',
                    y=column_to_plot,
                    title=f'{column_to_plot} sur la période de validation cumulée'
                )
                st.plotly_chart(fig_cumulative_performance)
                
                                        
        elif cross_val_type == "GridSearch":
            # Grid Search
            st.write("Définir les paramètres pour la Grid Search")
            param_grid = {
                "n_estimators": st.text_input("n_estimators (séparés par des virgules)", "50,100,200"),
                "max_depth": st.text_input("max_depth (séparés par des virgules)", "3,5,7"),
                "learning_rate": st.text_input("learning_rate (séparés par des virgules)", "0.01,0.1,0.2")
            }

            param_grid = {key: [float(v) if '.' in v else int(v) for v in value.split(',')] for key, value in param_grid.items()}

            metric = st.selectbox("Sélectionnez la métrique pour la Grid Search :", ['rmse', 'mae', 'smape'])

            if st.button("Lancer la Grid Search"):
                try:
                    best_params, grid_results = grid_search_xgboost(X, y, param_grid, metric=metric)
                    st.write("Meilleurs paramètres :", best_params)
                    st.dataframe(grid_results)

                    csv_grid = grid_results.to_csv(index=False)
                    st.download_button(
                        label="Télécharger les résultats de la Grid Search",
                        data=csv_grid,
                        file_name='grid_search_results.csv',
                        mime='text/csv'
                    )
                except Exception as e:
                    st.error(f"Erreur lors de la Grid Search : {e}")

    else:
        st.warning("Veuillez vérifier que les colonnes nécessaires (features et target) ont bien été définies dans la page précédente.")
else:
    st.warning("Aucune donnée filtrée trouvée. Veuillez retourner à la page précédente pour préparer les données.")
