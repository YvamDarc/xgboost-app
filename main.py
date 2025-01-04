import streamlit as st

def cover_page():
    # Configurer la page
    st.set_page_config(page_title="Application de Prévision XGBoost", layout="centered")

    # Ajouter une belle image
    st.image(
        "https://miro.medium.com/v2/resize:fit:827/1*txfLuX42exWSrKAzTJ3y5w.png", 
        caption="Transformez vos données en décisions stratégiques", 
        use_column_width=True
    )

    # Titre principal
    st.title("📘 Bienvenue dans l'application de Prévision XGBoost")

    # Résumé rapide
    st.markdown(
        """
        **Cette application innovante** vous permet d'exploiter vos données efficacement grâce à l'algorithme puissant 
        XGBoost. Vous pouvez importer vos fichiers, entraîner un modèle performant, visualiser les prévisions, et analyser 
        les écarts pour comprendre les facteurs influents.

        🌟 **Fonctionnalités clés** :
        - Importation simple de données Excel/CSV.
        - Entraînement intuitif avec personnalisation des paramètres.
        - Visualisation claire et analytique des prévisions.
        - Identification des causes des écarts pour une amélioration continue.

        🚀 **Prêt à commencer ? Explorez les fonctionnalités et transformez vos données en insights actionnables.**
        """
    )

if __name__ == "__main__":
    cover_page()