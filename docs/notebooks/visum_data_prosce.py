# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 17:20:28 2024

@author: Albert
"""

import scanpy
import pandas as pd
temp=pd.DataFrame()
a=scanpy.read_10x_mtx("../visium_data/sccanals26/BA2142")
df1=pd.read_csv("../visium_data/sccanals26/BA2142/barcodes.tsv",header=None).reset_index()
df1=df1.rename(columns={0: "barcode"})
df2=pd.read_csv("../visium_data/sccanals26/BA2142/features.tsv", sep=';',header=None).reset_index()
df2=df2.rename(columns={0: "gene id",1:"gene name"})
df2["gene name"]=df2["gene name"].apply(lambda x: x.replace("Gene Expression", "").strip())
df3=pd.read_csv("../visium_data/sccanals26/BA2142/matrix.mtx", sep=' ',header=2)
df3=df3.rename(columns={"4511660": "gene expresion"})
d13=pd.merge(df3, df1, left_on="2029",right_on="index")
d13 = d13.drop('2029', axis=1)
d13 = d13.drop('index', axis=1)
d13=pd.merge(d13, df2, left_on="32285",right_on="index", how="left")
d13 = d13.drop('32285', axis=1)
d13 = d13.drop('index', axis=1)
# ids=df2["gene id"].to_list()
# barcodes=df1["barcode"].to_list()
# for i in barcodes:
#     for j in ids:
#         tmp=d13[(d13["barcode"]==i) & (d13["gene id"]==j)]
#         if tmp.empty:
#             new_row = {'gene expresion': 0, 'barcode': i,'gene id': j, 'gene name': df2[df2["gene id"]==j]["gene name"].values[0]}
#             d13 = d13.append(new_row, ignore_index=True)
#             d13.loc[len(d13)] = new_row
            
resum=d13.pivot_table(index="barcode",aggfunc=list)
fname = '../visium_data/sccanals26/A1_tissue_positions.csv'
visium_points = pd.read_csv(fname)
visium_points[visium_points["in_tissue"]==1]["barcode"].unique().shape



