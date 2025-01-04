import streamlit as st

def main():
    st.title("📘 Manuel d'utilisation de l'application de prévision")
    
    st.header("🌟 Introduction à l'application de prévision basée sur XGBoost")
    
    st.markdown(
        """
        Cette application a été conçue pour répondre aux besoins des professionnels souhaitant exploiter les données 
        pour effectuer des prévisions précises et en analyser les écarts. En s'appuyant sur l'algorithme **XGBoost**, 
        reconnu pour ses performances dans les problèmes de régression et de classification, elle permet une prise de 
        décision éclairée et proactive.
        
        ---
        
        ## ⚙️ Fonctionnalités principales
        
        ### 1️⃣ Importation des données
        📂 Les utilisateurs peuvent charger facilement leurs données sous forme de tableaux Excel ou de fichiers CSV. 
        Cette flexibilité garantit une intégration rapide et adaptée à différents formats de travail.

        ### 2️⃣ Entraînement du modèle
        🧠 Après l'importation, une étape clé consiste à entraîner l'algorithme **XGBoost** sur les données fournies. L'utilisateur 
        a la possibilité de paramétrer le modèle afin d'optimiser les résultats en fonction des besoins spécifiques de chaque projet.

        ### 3️⃣ Visualisation et analyse des prévisions
        📊 Une fois le modèle entraîné, l'application génère des prévisions que l'utilisateur peut visualiser sous forme de graphiques 
        et de tableaux. Les écarts entre les prévisions et les valeurs réelles sont également mis en avant, offrant une première base d'analyse.

        ### 4️⃣ Recherche des causes des écarts
        🔍 Pour approfondir l'analyse, une page dédiée invite l'utilisateur à répondre à un questionnaire ciblé. Cet outil vise à identifier 
        les facteurs sous-jacents aux écarts observés, facilitant ainsi une amélioration continue des processus décisionnels.
        
        ---
        
        ## 🎯 Conclusion
        Avec cette structure, l'application propose une démarche complète, de la préparation des données à l'exploitation des résultats. 
        Elle se veut accessible aux utilisateurs de tous niveaux tout en intégrant des outils puissants pour répondre aux enjeux complexes des prévisions.
        """
    )

if __name__ == "__main__":
    main()
