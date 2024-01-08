#Importe de librerias

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline  
import os
import rasterio
import rasterio.features
import rasterio.warp
from osgeo import gdal
from matplotlib import pyplot
from rasterio.plot import show 

import geopandas as gpd
#Empezamos cargando librerías
from osgeo import ogr, osr

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,f1_score
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split 


os.chdir(r'C:\Users\giambelluca.144055\OneDrive\Doctorado\Datos_Laura')

dC = pd.read_csv('Claras_B7_B6_Prob_MAG.csv')
dCon = pd.read_csv('Control_B7_B6_Prob_MAG.csv')
dCH = pd.read_csv('Cortas_B7_B6_Prob_MAG.csv')


os.chdir(r'C:\Users\giambelluca.144055\OneDrive\Doctorado\Capas\Purgadas')
SHP_Claras=gpd.read_file('Claras_10102023.shp')
SHP_Control=gpd.read_file('Control_10102023.shp')
SHP_Cortas=gpd.read_file('Cortas_10102023.shp')

#Claras

union_claras=pd.merge(dC, SHP_Claras, on='ID')

union_claras['%2016']=(union_claras['P1_16']/(union_claras['N_pixel']))*100
union_claras['%2017']=(union_claras['P1_17']/(union_claras['N_pixel']))*100
union_claras['%2018']=(union_claras['P1_18']/(union_claras['N_pixel']))*100
union_claras['%2019']=(+union_claras['P1_19']/(union_claras['N_pixel']))*100
union_claras['%2020']=((union_claras['P1_20'])/(union_claras['N_pixel']))*100
union_claras['%2021']=((union_claras['P1_21'])/(union_claras['N_pixel']))*100
union_claras['%2022']=(union_claras['P1_22'])/(union_claras['N_pixel'])*100

union_claras['Suma_2016']=union_claras['%2016']
union_claras['Suma_2017']=union_claras['%2017']
union_claras['Suma_2018']=union_claras['%2018']
union_claras['Suma_2019']=union_claras['%2019']
union_claras['Suma_2020']=union_claras['%2020']
union_claras['Suma_2021']=union_claras['%2021']
union_claras['Suma_2022']=union_claras['%2022']
ac_claras=union_claras
ac_claras['Fecha_apro']=ac_claras['Fecha_apro'].astype(float)

#CH


union_cortas=pd.merge(dCH, SHP_Cortas, on='ID')


union_cortas['%2016']=(union_cortas['P1_16']/(union_cortas['N_pixel']))*100
union_cortas['%2017']=(union_cortas['P1_17']/(union_cortas['N_pixel']))*100
union_cortas['%2018']=(union_cortas['P1_18']/(union_cortas['N_pixel']))*100
union_cortas['%2019']=(union_cortas['P1_19']/(union_cortas['N_pixel']))*100
union_cortas['%2020']=((union_cortas['P1_20'])/(union_cortas['N_pixel']))*100
union_cortas['%2021']=((union_cortas['P1_21'])/(union_cortas['N_pixel']))*100
union_cortas['%2022']=(union_cortas['P1_22'])/(union_cortas['N_pixel'])*100

union_cortas['Suma_2016']=union_cortas['%2016']
union_cortas['Suma_2017']=union_cortas['%2017']
union_cortas['Suma_2018']=union_cortas['%2018']
union_cortas['Suma_2019']=union_cortas['%2019']
union_cortas['Suma_2020']=union_cortas['%2020']
union_cortas['Suma_2021']=union_cortas['%2021']
union_cortas['Suma_2022']=union_cortas['%2022']

ac_cortas=union_cortas

ac_cortas['Fecha_apro']=ac_cortas['Fecha_apro'].astype(float)


#Control

SHP_Control.columns = ['ID', 'Especie', 'Area', 'Area_Calcu', 'Perimeter', 'RATIO', 'Numero','ABREV','geometry']

union_control=pd.merge(dCon, SHP_Control, on='ID')


union_control['%2016']=(union_control['P1_16']/(union_control['N_pixel']))*100
union_control['%2017']=(union_control['P1_17']/(union_control['N_pixel']))*100
union_control['%2018']=(union_control['P1_18']/(union_control['N_pixel']))*100
union_control['%2019']=(union_control['P1_19']/(union_control['N_pixel']))*100
union_control['%2020']=((union_control['P1_20'])/(union_control['N_pixel']))*100
union_control['%2021']=((union_control['P1_21'])/(union_control['N_pixel']))*100
union_control['%2022']=(union_control['P1_22'])/(union_control['N_pixel'])*100

union_control['Suma_2016']=union_control['%2016']
union_control['Suma_2017']=union_control['%2017']
union_control['Suma_2018']=union_control['%2018']
union_control['Suma_2019']=union_control['%2019']
union_control['Suma_2020']=union_control['%2020']
union_control['Suma_2021']=union_control['%2021']
union_control['Suma_2022']=union_control['%2022']
union_control.columns
ac_control=union_control


cortas=pd.concat([selec_cortas,selec_CH])
claras=pd.concat([selec_C,selec_claras])
control=ac_control
cortas.loc[(cortas['ID'] == 268)|(cortas['ID'] == 269), ['Clara_Hech']] = ['H']

sel_cortas=cortas.sample(n = 220, random_state = 4)
sel_claras=claras.sample(n = 220, random_state = 4)
sel_control=control.sample(n = 220, random_state = 4)

np.random.seed(1)

random_claras, random_claras_test = train_test_split(sel_claras, test_size = 0.30)

np.random.seed(1)

random_cortas, random_cortas_test = train_test_split(sel_cortas, test_size = 0.30)

np.random.seed(1)

random_control, random_control_test = train_test_split(sel_control, test_size = 0.30)

unido = pd.concat([random_claras,random_cortas, random_control])
# unido = pd.concat([cortas,claras, control])
unido['Clara_Hech']=unido['Clara_Hech'].fillna('N')

umbral_bajo=[]
umbral_alto=[]
mag_baja=[]
mag_alta=[]
f_score_micro=[]
f_score_macro=[]
accuracy=[]

for c in range (1000,-1,-10):
        for d in range (10,1001,10):
            a=c/10000
            b=d/10000
            if a>b:
#                     aprov.loc[(aprov['M7_16']>=b)&(aprov['M7_16']<a), ['2016']] = ['C']
#                     aprov.loc[(aprov['M7_16']<b), ['2016']] = ['N']
#                     aprov.loc[(aprov['M7_16']>a), ['2016']] = ['H']


#                     aprov.loc[(aprov['M7_17']>=b)&(aprov['M7_17']<a), ['2017']] = ['C']
#                     aprov.loc[(aprov['M7_17']<b), ['2017']] = ['N']
#                     aprov.loc[(aprov['M7_17']>a), ['2017']] = ['H']

#                     aprov.loc[(aprov['M7_18']>=b)&(aprov['M7_18']<a), ['2018']] = ['C']
#                     aprov.loc[(aprov['M7_18']<b), ['2018']] = ['N']
#                     aprov.loc[(aprov['M7_18']>a), ['2018']] = ['H']

#                     aprov.loc[(aprov['M7_19']>=b)&(aprov['M7_19']<a), ['2019']] = ['C']
#                     aprov.loc[(aprov['M7_19']<b), ['2019']] = ['N']
#                     aprov.loc[(aprov['M7_19']>a), ['2019']] = ['H']

#                     aprov.loc[(aprov['M7_20']>=b)&(aprov['M7_20']<a), ['2020']] = ['C']
#                     aprov.loc[(aprov['M7_20']<b), ['2020']] = ['N']
#                     aprov.loc[(aprov['M7_20']>a), ['2020']] = ['H']


#                     aprov.loc[(aprov['M7_21']>=b)&(aprov['M7_21']<a), ['2021']] = ['C']
#                     aprov.loc[(aprov['M7_21']<b), ['2021']] = ['N']
#                     aprov.loc[(aprov['M7_21']>a), ['2021']] = ['H']

#                     aprov.loc[(aprov['M7_22']>=b)&(aprov['M7_22']<a), ['2022']] = ['C']
#                     aprov.loc[(aprov['M7_22']<b), ['2022']] = ['N']
#                     aprov.loc[(aprov['M7_22']>a), ['2022']] = ['H']
                    aprov.loc[(aprov['Mb6_16']>=b)&(aprov['Mb6_16']<a), ['2016']] = ['C']
                    aprov.loc[(aprov['Mb6_16']<b), ['2016']] = ['N']
                    aprov.loc[(aprov['Mb6_16']>a), ['2016']] = ['H']


                    aprov.loc[(aprov['Mb6_17']>=b)&(aprov['Mb6_17']<a), ['2017']] = ['C']
                    aprov.loc[(aprov['Mb6_17']<b), ['2017']] = ['N']
                    aprov.loc[(aprov['Mb6_17']>a), ['2017']] = ['H']

                    aprov.loc[(aprov['Mb6_18']>=b)&(aprov['Mb6_18']<a), ['2018']] = ['C']
                    aprov.loc[(aprov['Mb6_18']<b), ['2018']] = ['N']
                    aprov.loc[(aprov['Mb6_18']>a), ['2018']] = ['H']

                    aprov.loc[(aprov['Mb6_19']>=b)&(aprov['Mb6_19']<a), ['2019']] = ['C']
                    aprov.loc[(aprov['Mb6_19']<b), ['2019']] = ['N']
                    aprov.loc[(aprov['Mb6_19']>a), ['2019']] = ['H']

                    aprov.loc[(aprov['Mb6_20']>=b)&(aprov['Mb6_20']<a), ['2020']] = ['C']
                    aprov.loc[(aprov['Mb6_20']<b), ['2020']] = ['N']
                    aprov.loc[(aprov['Mb6_20']>a), ['2020']] = ['H']


                    aprov.loc[(aprov['Mb6_21']>=b)&(aprov['Mb6_21']<a), ['2021']] = ['C']
                    aprov.loc[(aprov['Mb6_21']<b), ['2021']] = ['N']
                    aprov.loc[(aprov['Mb6_21']>a), ['2021']] = ['H']

                    aprov.loc[(aprov['Mb6_22']>=b)&(aprov['Mb6_22']<a), ['2022']] = ['C']
                    aprov.loc[(aprov['Mb6_22']<b), ['2022']] = ['N']
                    aprov.loc[(aprov['Mb6_22']>a), ['2022']] = ['H']

                    aprov['verif']='F'
                    aprov['2023']='X'
                    aprov['2024']='X'

                    year=['2016','2017','2018','2019','2020','2021','2022','2023','2024']
                    for y in range(len(year)):
                        if year[y] != '2023':
                            aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='C')&(aprov[year[y+1]] =='C')&(aprov['verif'] !='T'), ['verif']] = ['T']
                            aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='C')&(aprov[year[y]] =='C')&(aprov['verif'] !='T'), ['verif']] = ['T']
                            aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='C')&(aprov[year[y+2]] =='C')&(aprov['verif'] !='T'), ['verif']] = ['T']

                            aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='H')&(aprov[year[y+1]] =='H')&(aprov['verif'] !='T'), ['verif']] = ['T']
                            aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='H')&(aprov[year[y]] =='H')&(aprov['verif'] !='T'), ['verif']] = ['T']
                            aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='H')&(aprov[year[y+2]] =='H')&(aprov['verif'] !='T'), ['verif']] = ['T']


                        else:
                            break


                    #Si en todos los años hay control

                    aprov.loc[(aprov['Clara_Hech']=='N')&(aprov['2018'] =='N')&(aprov['2019'] =='N')&(aprov['2020'] =='N')&(aprov['2021'] =='N'), ['verif']] = ['T']

                    aprov['pred']='X'

                    aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='H'), ['pred']] = ['H']
                    aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='C'), ['pred']] = ['C']
                    aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='N'), ['pred']] = ['N']



                    #Falsos coloca la clasificacion que tiene esa fecha
                    year=['2016','2017','2018','2019','2020','2021','2022','2023']
                    for i in range(len(year)):
                            if i<5:
                                aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
                                aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='C'),['pred']] = ['C']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i+2]]=='C'),['pred']] = ['C']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='H'),['pred']] = ['H']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i+2]]=='H'),['pred']] = ['H']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&((aprov[year[i]]=='N')&(aprov[year[i+2]]=='N')),['pred']] = ['N']

                            elif year[i]=='2021':
                                aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
                                aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='C'),['pred']] = ['C']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='H'),['pred']] = ['H']
                                aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='N'),['pred']] = ['N']

                            else:
                                break


                    #Falsos de control coloca la primera ruptura
                    year=['2018','2019','2020','2021']
                    for h in range(len(year)):
                            aprov.loc[(aprov['verif'] =='F')&(aprov['Clara_Hech']=='N')&(aprov[year[h]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
                            aprov.loc[(aprov['verif'] =='F')&(aprov['Clara_Hech']=='N')&(aprov[year[h]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']

                    y_aprov=np.array(aprov['pred'])
                    y_true_aprov=np.array(aprov['Clara_Hech'])
                    f_score_mic=f1_score(y_true_aprov, y_aprov, labels=['C','H','N'], average='micro', sample_weight=None, zero_division='warn')
                    f_score_mac=f1_score(y_true_aprov, y_aprov, labels=['C','H','N'], average='macro', sample_weight=None, zero_division='warn')
            
                    accuracy_x=accuracy_score(y_true_aprov, y_aprov)
                    f_score_micro.append(f_score_mic)
                    f_score_macro.append(f_score_mac)
                    mag_alta.append(a)
                    mag_baja.append(b)
                    accuracy.append(accuracy_x)

                    print(c,d)


analisis= pd.DataFrame()
analisis['Magnitud_alta']=mag_alta
analisis['Magnitud_baja']=mag_baja
analisis['F-Score_Macro']=f_score_macro
analisis['F-Score_Micro']=f_score_micro
analisis['Accuracy']=accuracy
analisis.loc[(analisis['F-Score_Macro']==analisis['F-Score_Macro'].max())]


unido = pd.concat([random_claras_test,random_cortas_test, random_control_test])
unido['Clara_Hech']=unido['Clara_Hech'].fillna('N')
aprov=unido


#Ver
a=0.034
b=0.001
x=3
i=30

aprov.loc[(aprov['M7_16']>=b)&(aprov['M7_16']<a), ['2016']] = ['C']
aprov.loc[(aprov['M7_16']<b), ['2016']] = ['N']
aprov.loc[(aprov['M7_16']>a), ['2016']] = ['H']

aprov.loc[(aprov['M7_17']>=b)&(aprov['M7_17']<a), ['2017']] = ['C']
aprov.loc[(aprov['M7_17']<b), ['2017']] = ['N']
aprov.loc[(aprov['M7_17']>a), ['2017']] = ['H']

aprov.loc[(aprov['M7_18']>=b)&(aprov['M7_18']<a), ['2018']] = ['C']
aprov.loc[(aprov['M7_18']<b), ['2018']] = ['N']
aprov.loc[(aprov['M7_18']>a), ['2018']] = ['H']

aprov.loc[(aprov['M7_19']>=b)&(aprov['M7_19']<a), ['2019']] = ['C']
aprov.loc[(aprov['M7_19']<b), ['2019']] = ['N']
aprov.loc[(aprov['M7_19']>a), ['2019']] = ['H']

aprov.loc[(aprov['M7_20']>=b)&(aprov['M7_20']<a), ['2020']] = ['C']
aprov.loc[(aprov['M7_20']<b), ['2020']] = ['N']
aprov.loc[(aprov['M7_20']>a), ['2020']] = ['H']

aprov.loc[(aprov['M7_21']>=b)&(aprov['M7_21']<a), ['2021']] = ['C']
aprov.loc[(aprov['M7_21']<b), ['2021']] = ['N']
aprov.loc[(aprov['M7_21']>a), ['2021']] = ['H']

aprov.loc[(aprov['M7_22']>=b)&(aprov['M7_22']<a), ['2022']] = ['C']
aprov.loc[(aprov['M7_22']<b), ['2022']] = ['N']
aprov.loc[(aprov['M7_22']>a), ['2022']] = ['H']

aprov['2016']=aprov['2016'].fillna('N')
aprov['2017']=aprov['2017'].fillna('N')
aprov['2018']=aprov['2018'].fillna('N')
aprov['2019']=aprov['2019'].fillna('N')
aprov['2020']=aprov['2020'].fillna('N')
aprov['2021']=aprov['2021'].fillna('N')
aprov['2022']=aprov['2022'].fillna('N')
                            
aprov['verif']='F'
aprov['2023']='X'
aprov['2024']='X'

year=['2016','2017','2018','2019','2020','2021','2022','2023','2024']
for y in range(len(year)):
        if year[y] != '2023':
                aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='C')&(aprov[year[y+1]] =='C')&(aprov['verif'] !='T'), ['verif']] = ['T']
                aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='C')&(aprov[year[y]] =='C')&(aprov['verif'] !='T'), ['verif']] = ['T']
                aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='C')&(aprov[year[y+2]] =='C')&(aprov['verif'] !='T'), ['verif']] = ['T']

                aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='H')&(aprov[year[y+1]] =='H')&(aprov['verif'] !='T'), ['verif']] = ['T']
                aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='H')&(aprov[year[y]] =='H')&(aprov['verif'] !='T'), ['verif']] = ['T']
                aprov.loc[(aprov['Fecha_apro'] ==int(year[y+1]))&(aprov['Clara_Hech'] =='H')&(aprov[year[y+2]] =='H')&(aprov['verif'] !='T'), ['verif']] = ['T']


        else:
            break


#Si en todos los años hay control

aprov.loc[(aprov['Clara_Hech']=='N')&(aprov['2018'] =='N')&(aprov['2019'] =='N')&(aprov['2020'] =='N')&(aprov['2021'] =='N'), ['verif']] = ['T']

aprov['pred']='X'

aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='H'), ['pred']] = ['H']
aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='C'), ['pred']] = ['C']
aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='N'), ['pred']] = ['N']



#Falsos coloca la clasificacion que tiene esa fecha
year=['2016','2017','2018','2019','2020','2021','2022','2023']
for i in range(len(year)):
    if i<5:
        aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
        aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='C'),['pred']] = ['C']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i+2]]=='C'),['pred']] = ['C']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='H'),['pred']] = ['H']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i+2]]=='H'),['pred']] = ['H']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&((aprov[year[i]]=='N')&(aprov[year[i+2]]=='N')),['pred']] = ['N']

    elif year[i]=='2021':
        aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
        aprov.loc[(aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='C'),['pred']] = ['C']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='H'),['pred']] = ['H']
        aprov.loc[((aprov['Fecha_apro'] ==int(year[i+1]))&(aprov['verif'] =='F')&(aprov[year[i+1]]=='N')&(aprov['pred']=='X'))&(aprov[year[i]]=='N'),['pred']] = ['N']

    else:
        break


#Falsos de control coloca la primera ruptura
year=['2018','2019','2020','2021']
for h in range(len(year)):
        aprov.loc[(aprov['verif'] =='F')&(aprov['Clara_Hech']=='N')&(aprov[year[h]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
        aprov.loc[(aprov['verif'] =='F')&(aprov['Clara_Hech']=='N')&(aprov[year[h]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']


y_aprov=np.array(aprov['pred'])
y_true_aprov=np.array(aprov['Clara_Hech'])

confusion_matrix(y_true_aprov, y_aprov, labels=['C','H','N'])
