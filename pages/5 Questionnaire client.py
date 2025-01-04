import streamlit as st
import pandas as pd
from datetime import date
import os

# Chemin du fichier Excel d'archivage
FILE_PATH = "analyses_ecarts.xlsx"

def save_to_excel(data):
    if not os.path.exists(FILE_PATH):
        # Créer un nouveau fichier si n'existe pas
        df = pd.DataFrame(data)
        df.to_excel(FILE_PATH, index=False)
    else:
        # Charger le fichier existant et ajouter les nouvelles données
        existing_df = pd.read_excel(FILE_PATH)
        new_df = pd.DataFrame(data)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_excel(FILE_PATH, index=False)

def main():
    st.title("Analyse des Causes des Écarts")
    st.header("Identifiez les causes des variations dans le chiffre d'affaires de votre entreprise")

    # Sélection de la date et de l'écart
    st.subheader("Date et écart de l'analyse")
    analysis_date = st.date_input("Sélectionnez une date", value=date.today(), key="analysis_date")
    analysis_gap = st.text_input("Renseignez l'écart (en pourcentage ou valeur absolue, positif ou négatif)", key="analysis_gap")

    # Causes externes
    st.subheader("Causes externes")

    st.markdown("### Demande des clients et facteurs externes")
    external_demand_notes = []
    external_demand_questions = [
        ("Apparition de nouveaux acteurs influençant le marché", "detail_new_competitors"),
        ("Évolution des besoins ou préférences des clients", "detail_changing_needs"),
        ("Contexte économique influençant les comportements d'achat", "detail_economic_context"),
        ("Variations saisonnières affectant la demande", "detail_seasonal_variations"),
        ("Réglementations impactant la consommation", "detail_regulations"),
        ("Disponibilité ou coût des matières premières", "detail_materials_availability"),
        ("Perturbations logistiques ou d'approvisionnement", "detail_logistics"),
        ("Image de marque","detail_image"),
        ("Autre (facteurs externes)", "detail_other_external")
    ]

    for question, key in external_demand_questions:
        if st.checkbox(question, key=key):
            note = st.text_input(f"Détail pour '{question}'", key=f"{key}_note")
            external_demand_notes.append(f"- {question} : {note}")

    st.markdown("### Publicité et communication")
    external_publicity_notes = []
    external_publicity_questions = [
        ("Publicité et communication inadaptées", "detail_poor_publicity"),
        ("Absence ou inefficacité des campagnes marketing", "detail_marketing_campaigns"),
        ("Avis clients ou bouche-à-oreille influençant la réputation", "detail_customer_feedback"),
        ("Autre (publicité et communication)", "detail_other_publicity")
    ]

    for question, key in external_publicity_questions:
        if st.checkbox(question, key=key):
            note = st.text_input(f"Détail pour '{question}'", key=f"{key}_note")
            external_publicity_notes.append(f"- {question} : {note}")

    # Causes internes
    st.subheader("Causes internes")

    st.markdown("### Offre produit et méthode de production")
    internal_product_notes = []
    internal_product_questions = [
        ("Adaptation des produits ou services aux attentes des clients", "detail_product_adaptation"),
        ("Performance ou fiabilité des produits ou services", "detail_product_performance"),
        ("Optimisation des coûts de production", "detail_production_costs"),
        ("Gestion des stocks (ruptures ou excès)", "detail_stock_management"),
        ("Méthodes de production ou d'exécution modernes et efficaces", "detail_production_methods"),
        ("Innovation et développement de nouvelles offres", "detail_innovation"),
        ("Autre (offre produit et méthode de production)", "detail_other_internal")
    ]

    for question, key in internal_product_questions:
        if st.checkbox(question, key=key):
            note = st.text_input(f"Détail pour '{question}'", key=f"{key}_note")
            internal_product_notes.append(f"- {question} : {note}")

    st.markdown("### Organisation et personnel")
    internal_personnel_notes = []
    internal_personnel_questions = [
        ("Coordination entre services internes", "detail_internal_coordination"),
        ("Engagement et satisfaction des collaborateurs", "detail_employee_engagement"),
        ("Absences ou fluctuations dans la main-d'œuvre", "detail_workforce_fluctuations"),
        ("Processus internes (suivi des tâches, objectifs)", "detail_internal_processes"),
        ("Autre (organisation et personnel)", "detail_other_personnel")
    ]

    for question, key in internal_personnel_questions:
        if st.checkbox(question, key=key):
            note = st.text_input(f"Détail pour '{question}'", key=f"{key}_note")
            internal_personnel_notes.append(f"- {question} : {note}")

    # Résumé de l'analyse
    st.subheader("Résumé de votre analyse")
    export_data = []

    for subheader, notes in zip(
        ["Causes externes - Demande et facteurs", "Causes externes - Publicité et communication", "Causes internes - Offre produit et méthode de production", "Causes internes - Organisation et personnel"],
        [external_demand_notes, external_publicity_notes, internal_product_notes, internal_personnel_notes]
    ):
        for note in notes:
            export_data.append({
                "Date": analysis_date,
                "Écart": analysis_gap,
                "Sous-section": subheader,
                "Détail": note
            })

    if export_data:
        for record in export_data:
            st.write(f"{record['Sous-section']} : {record['Détail']}")

        if st.button("Enregistrer l'analyse", key="save_analysis"):
            try:
                save_to_excel(export_data)
                st.success("L'analyse a été enregistrée avec succès dans le fichier Excel !")
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement : {e}")
    else:
        st.info("Aucune donnée à enregistrer. Veuillez effectuer votre analyse avant de sauvegarder.")

if __name__ == "__main__":
    main()
