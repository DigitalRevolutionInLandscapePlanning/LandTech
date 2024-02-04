# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 14:27:46 2023

@author: DigitalRevolutionInLandscapePlanning

python script for generating correlation charts
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

plt.rcParams.update({'figure.dpi':300})

#df = pd.read_excel('C:/ProjektyPython/textClasification/output/ALL PAPERS MAY10_processed_with_pred_app32092023_2.xlsx')
df = pd.read_excel('C:/ProjektyPython/textClasification/output/ALL PAPERS MAY10_processed_with_pred_app_12112023.xlsx')

years = df["Year"].unique().tolist()


def generate_corr_matrix_and_list_of_articles(columns, column_names,title):
    df2 = df[columns]
    df2.columns = column_names
    corr_matrix = df2.corr().round(2)
    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax)
    ax.set_title(title)
    plt.show()
    for index,row in corr_matrix.iterrows():
        for col in corr_matrix.columns:
            if 0.1 <= corr_matrix[index][col] < 1:
                iname= index.replace("/","")
                cname=col.replace("/","")
                #df.iloc[:,:53][(df[cols[col_names.index(index)]]==1) & (df[cols[col_names.index(col)]]==1)].to_csv('C:/ProjektyPython/textClasification/output/listy/cor_{}_{}_{}.csv'.format(iname,cname,corr_matrix[index][col]), index = True, sep='\t')


cols = ["Data","VR/AR/MR","all3dmodeling","AI&related",
                "MCDA/PSS","allICT","Robotic","GIS","Remote sensing",
                "Other Digital Modelling","Social Media","CAD"]
col_names = ["Data","VR/AR/MR","3D Modelling","AI Related",
               "Decision Support","Other ICT", "Robotic","GIS","Remote Sensing",
               "Other Digital Modelling","Social Media","CAD"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total')


for year in years:
    cols = ["Data","VR/AR/MR","all3dmodeling","AI&related",
                    "MCDA/PSS","allICT","Robotic","GIS","Remote sensing",
                    "Other Digital Modelling","Social Media","CAD"]
    col_names = ["Data","VR/AR/MR","3D Modelling","AI Related",
                   "Decision Support","Other ICT", "Robotic","GIS","Remote Sensing",
                   "Other Digital Modelling","Social Media","CAD"]
    df2 = df[df['Year']==year][cols]
    df2.columns = col_names
    
    corr_matrix = df2.corr().round(2)
    sns.set(style='white')
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', vmin=-1, vmax=1, ax=ax)
    ax.set_title(str(year))
    plt.show()
    for index,row in corr_matrix.iterrows():
        for col in corr_matrix.columns:
            if 0.1 <= corr_matrix[index][col] < 1:
                iname= index.replace("/","")
                cname=col.replace("/","")
                df.iloc[:,:53][(df[cols[col_names.index(index)]]==1) & (df[cols[col_names.index(col)]]==1)].to_csv('C:/ProjektyPython/textClasification/output/listy/cor_{}_{}_{}_{}.csv'.format(year,iname,cname,corr_matrix[index][col]), index = True, sep='\t')



#Data – Remote Sensing
cols = ['Data science','Data mining','Big data','Crowdsourcing',
                'Open data', 'OthersData',"Remote sensing"]
col_names = ['Data Science','Data Mining','Big Data','Crowdsourcing',
                'Open Data', 'Other Data',"Remote Sensing"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Data – Remote Sensing')

#Data – Social Media
cols = ['Data science','Data mining','Big data','Crowdsourcing',
                'Open data', 'OthersData',"Social Media"]
col_names = ['Data Science','Data Mining','Big Data','Crowdsourcing',
                'Open Data', 'Other Data',"Social Media"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Data – Social Media')

#Data – AI Related
cols = ['Data science','Data mining','Big data','Crowdsourcing',
                'Open data', 'OthersData',"Deep Learning", "Artificial Intelligence", "Machine learning", "MAS", 'OthersAI']
col_names = ['Data Science','Data Mining','Big Data','Crowdsourcing',
                'Open Data', 'Other Data',"Deep Learning", "Artificial Intelligence", "Machine Learning", "MAS", 'Other AI']

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Data – AI Related')

#Data – Other ICT
cols=['Data science','Data mining','Big data','Crowdsourcing',
                'Open data', 'OthersData',"GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices"]
col_names = ['Data Science','Data Mining','Big Data','Crowdsourcing',
                'Open Data', 'Other Data',"GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Data – Other ICT')

#VR/AR/MR – 3d Modelling
cols=["VR", "AR", "MR","BIM", "LIM", "PointCloud", "Rendering", "3dScanning", "3dModelling"]
col_names = ["VR", "AR", "MR","BIM", "LIM", "Point Cloud", "Rendering", "3D Scanning", "3D Modelling"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: VR/AR/MR – 3d Modelling')


#3d Modelling – Other ICT
cols = ["BIM", "LIM", "PointCloud", "Rendering", "3dScanning", "3dModelling","GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices"]
col_names = ["BIM", "LIM", "Point Cloud", "Rendering", "3D Scanning", "3D Modelling","GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: 3d Modelling – Other ICT')

#3d Modelling – Remote Sensing
cols = ["BIM", "LIM", "PointCloud", "Rendering", "3dScanning", "3dModelling","Remote sensing"]
col_names = ["BIM", "LIM", "Point Cloud", "Rendering", "3D Scanning", "3D Modelling","Remote Sensing"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: 3d Modelling – Remote Sensing')

#3d Modelling – Robotic
cols=["BIM", "LIM", "PointCloud", "Rendering", "3dScanning", "3dModelling","Robotic"]
col_names = ["BIM", "LIM", "Point Cloud", "Rendering", "3D Scanning", "3D Modelling","Robotic"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: 3d Modelling – Robotic')

#AI Related – Remote Sensing
cols=["Deep Learning", "Artificial Intelligence", "Machine learning", "MAS", "OthersAI", "Remote sensing"]
col_names = ["Deep Learning", "Artificial Intelligence", "Machine Learning", "MAS", "Other AI","Remote Sensing"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: AI Related – Remote Sensing')

#Decision support – GIS
cols = ["MCDA/AHP", "PSS/DSS","GIS"]
col_names = ["MCDA/AHP", "PSS/DSS","GIS"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Decision support – GIS')


#Other ICT – Remote Sensing
cols = ["GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices","Remote sensing"]
col_names = ["GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices","Remote Sensing"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Other ICT – Remote Sensing')


#Other ICT – Robotic
cols=["GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices","Robotic"]
col_names=["GPS", "GNSS", "ICT", "Blockchain", "Cloud", "IoT", "Devices","Robotic"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: Other ICT – Robotic')

#CAD – VR
cols=["CAD","VR", "AR", "MR"]
col_names = ["CAD","VR", "AR", "MR"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: CAD - VR/AR/MR')

#3d Modelling – CAD
cols=["BIM", "LIM", "PointCloud", "Rendering", "3dScanning", "3dModelling","CAD"]
col_names = ["BIM", "LIM", "Point Cloud", "Rendering", "3D Scanning", "3D Modelling","CAD"]

generate_corr_matrix_and_list_of_articles(cols, col_names,'Total: 3d Modelling – CAD')
