from altair import selection_interval
import streamlit as st
import pandas as pd
from pycaret.clustering import *
import plotly.io as pio
import plotly.express as px
#import "functions.py"
@st.cache
@st.cache_data
def long_running_function(param1, param2):
    return "…"



def load_data(file, formats):
  if formats == "csv":
    data = pd.read_csv(file, sep=",")
  else :
      data = pd.read_excel(file)
  return data


st.title("Application de clustering")


options = ["K-Means Clustering", 
    "Affinity Propagation",
    "Mean Shift Clustering",
    "Spectral Clustering",
    "Agglomerative Clustering",
    "Density-Based Spatial Clustering",
    "OPTICS Clustering","Birch Clustering	"]

list_models = {"K-Means Clustering" : "kmeans" , 
            "Affinity Propagation":"ap" ,
            "Mean Shift Clustering":"meanshift"  ,
            "Spectral Clustering":"sc"  ,
            "Agglomerative Clustering	":"hclust" ,
            "Density-Based Spatial Clustering":"dbscan" ,
            "OPTICS Clustering":"optics"  ,
            "Birch Clustering":"birch"  
            }



formats = ["csv","xlsx"]

selected_format = st.radio('Format', formats)

file = st.file_uploader("Importer vos données ici", type=[selected_format])

if file is not None:
  data = load_data(file, selected_format)
  print_df = st.radio("Afficher les données",["Non","Oui"])
  if print_df == "Oui":
    st.dataframe(data)
  columns_to_use = []
  columns_to_use = st.multiselect("Choisir les colonnes à utiliser pour le clustering",data.columns)
  if columns_to_use != []: 
    df = data[columns_to_use]
    if st.radio("Afficher",["Non","Oui"]) == "Oui":
      st.dataframe(df)
    

    methodes_pca = {
      'Linear': 'linear',
      'Kernel': 'kernel',
      'Incremental': 'incremental',
      'Sparse': 'sparse',
      'Truncated SVD': 'truncated_svd',
      'Exact': 'exact',
      'Full': 'full'
        } 
    methods = ['Linear','Kernel','Incremental','Sparse','Truncated SVD',
                  'Exact','Full']
    

    col_date = []
    if st.radio("Colone date ",["Non","Oui"]) == "Oui":
      col_date = st.multiselect("",columns_to_use)
      col_date = pd.to_datetime(col_date)
    dec_method = st.sidebar.selectbox("Méthode PCA",methods)
    setup_clustering = setup(data = df,normalize=True, session_id=123,pca=True, date_features=col_date,  pca_method=methodes_pca[dec_method] )
    setup_clustering_df = pull()
    if st.radio("Afficher setup",["Non","Oui"])=="Oui" : 
      st.dataframe(setup_clustering_df)


    selected_option = st.sidebar.radio('Algorithme de clustering:', options)
    if st.sidebar.selectbox("Nombre de cluster",["Automatique", "Spécifier soi-même"]) == "Spécifier soi-même":
      n_clusters = int(st.sidebar.slider("", min_value=2, max_value=10))

      clustering_model = create_model(list_models[selected_option],n_clusters)
      clustering_model_df = pull()
      st.dataframe(clustering_model_df)
    else : 
      clustering_model = create_model(list_models[selected_option])
      clustering_model_df = pull()
      st.write("Rapport de segmentation")
      st.dataframe(clustering_model_df)
    st.subheader("Clustering Plots")
    cluster_plot_options = ['Cluster PCA Plot (2d)','Cluster t-SNE (3d)',
                                'Elbow Plot','Silhouette Plot','Distance Plot','Distribution Plot']
    cluster_plot_options_dic = {'Cluster PCA Plot (2d)' : 'cluster',
                                  'Cluster t-SNE (3d)' : 'tsne',
                                'Elbow Plot' : 'elblow',
                                'Silhouette Plot' : 'silhouette',
                                'Distance Plot' : 'distance',
                                'Distribution Plot' : 'distribution'
                                }
    plot_choice = st.sidebar.radio("Choice a plot",cluster_plot_options)
    plot_model(clustering_model,cluster_plot_options_dic[plot_choice], display_format='streamlit' )


    if st.selectbox("Assigner les clusters aux données", ["Non","Oui"] ) == "Oui" :
      clustered_data = assign_model(model=clustering_model)
      st.dataframe(clustered_data)

      st.subheader("Résumé statistique de la segmentation")
      to_summarise = []
      to_summarise = st.multiselect("Choisir les variables en fonctions desquelles afficher le résumé", clustered_data.columns)
      if to_summarise != [] :
        sumarised_data = clustered_data.groupby(to_summarise)[clustered_data.drop(to_summarise, axis =1).columns].mean()
        st.dataframe(sumarised_data)

        if st.selectbox("Enregistrer le résumé", ["Non","Oui"]) == "Oui" :
          if st.radio("Format du fichier",["Xlsx","Csv"]) == "Csv" : 
            sumarised_data.to_csv("Résumé segmentation.csv")
          else :
            sumarised_data.to_excel("Résumé segmentation.xlsx")

      if st.selectbox("Enregistrer les données segmentées", ["Non","Oui"]) == "Oui" :
        if st.radio("Format du fichier",["Xlsx ","Csv "]) == "Csv" : 
          clustered_data.to_csv("Données segmenattion.csv")
        else :
          clustered_data.to_excel("Données segmenattion.xlsx")


      

      



