# Code to obtain the optimal threshold is you used magnitude of SWIR2 or SWIR1 as variable

#Import libraries

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
from osgeo import ogr, osr
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay, classification_report,f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split 

# First part: upload CSV and vectorial layers

direct='Path'
os.chdir(direct)

dC = pd.read_csv('Thinning_B7_B6.csv')
dCon = pd.read_csv('Control_B7_B6.csv')
dCH = pd.read_csv('CC_B7_B6.csv')

direct_vect='Path'
os.chdir(direct_vect)
SHP_T=gpd.read_file('Thinning_plot.shp')
SHP_Control=gpd.read_file('Control_plot.shp')
SHP_CC=gpd.read_file('Clear_cutting_plot.shp')

# Second part: join vectorial layer with CSV information through ID column.
#Thinning

join_T=pd.merge(dC, SHP_T, on='ID')
join_T['Fecha_apro']=join_T['Fecha_apro'].astype(float)

#Clear-cutting

join_CC=pd.merge(dCH, SHP_CC, on='ID')
join_CC['Fecha_apro']=join_CC['Fecha_apro'].astype(float)

#Control

SHP_Control.columns = ['ID', 'Especie', 'Area', 'Area_Calcu', 'Perimeter', 'RATIO', 'Numero','ABREV','geometry']
join_control=pd.merge(dCon, SHP_Control, on='ID')

# Third part: Select randomly 220 plot for each forest harvesting practice
sel_CC=join_CC.sample(n = 220, random_state = 4)
sel_T=join_T.sample(n = 220, random_state = 4)
sel_control=join_control.sample(n = 220, random_state = 4)

# Fourth part: Split samples in train (70%) and test (30%)
np.random.seed(1)
random_T, random_T_test = train_test_split(sel_T, test_size = 0.30)
np.random.seed(1)
random_CC, random_CC_test = train_test_split(sel_CC, test_size = 0.30)
np.random.seed(1)
random_control, random_control_test = train_test_split(sel_control, test_size = 0.30)

#Join the train plots
aprov = pd.concat([random_T,random_CC, random_control])
aprov['Clara_Hech']=aprov['Clara_Hech'].fillna('N')

#Fifth part: loop to know the optimal threshold

mag_low=[]
mag_high=[]
f_score_micro=[]
f_score_macro=[]
accuracy=[]

# Depend on the band considered the threshold might change. In this case is for SWIR2 or SWIR1
for c in range (1000,-1,-10):
        for d in range (10,1001,10):
            a=c/10000
            b=d/10000
            if a>b:
                   # In this example SWIR2 magnitude will be tested, for SWIR1 comment the following code and uncomment the next.
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

                    #SWIR1
                    # aprov.loc[(aprov['Mb6_16']>=b)&(aprov['Mb6_16']<a), ['2016']] = ['C']
                    # aprov.loc[(aprov['Mb6_16']<b), ['2016']] = ['N']
                    # aprov.loc[(aprov['Mb6_16']>a), ['2016']] = ['H']
                    # aprov.loc[(aprov['Mb6_17']>=b)&(aprov['Mb6_17']<a), ['2017']] = ['C']
                    # aprov.loc[(aprov['Mb6_17']<b), ['2017']] = ['N']
                    # aprov.loc[(aprov['Mb6_17']>a), ['2017']] = ['H']
                    # aprov.loc[(aprov['Mb6_18']>=b)&(aprov['Mb6_18']<a), ['2018']] = ['C']
                    # aprov.loc[(aprov['Mb6_18']<b), ['2018']] = ['N']
                    # aprov.loc[(aprov['Mb6_18']>a), ['2018']] = ['H']
                    # aprov.loc[(aprov['Mb6_19']>=b)&(aprov['Mb6_19']<a), ['2019']] = ['C']
                    # aprov.loc[(aprov['Mb6_19']<b), ['2019']] = ['N']
                    # aprov.loc[(aprov['Mb6_19']>a), ['2019']] = ['H']
                    # aprov.loc[(aprov['Mb6_20']>=b)&(aprov['Mb6_20']<a), ['2020']] = ['C']
                    # aprov.loc[(aprov['Mb6_20']<b), ['2020']] = ['N']
                    # aprov.loc[(aprov['Mb6_20']>a), ['2020']] = ['H']
                    # aprov.loc[(aprov['Mb6_21']>=b)&(aprov['Mb6_21']<a), ['2021']] = ['C']
                    # aprov.loc[(aprov['Mb6_21']<b), ['2021']] = ['N']
                    # aprov.loc[(aprov['Mb6_21']>a), ['2021']] = ['H']
                    # aprov.loc[(aprov['Mb6_22']>=b)&(aprov['Mb6_22']<a), ['2022']] = ['C']
                    # aprov.loc[(aprov['Mb6_22']<b), ['2022']] = ['N']
                    # aprov.loc[(aprov['Mb6_22']>a), ['2022']] = ['H']

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
                    mag_high.append(a)
                    mag_low.append(b)
                    accuracy.append(accuracy_x)

                    print(c,d)


analisis= pd.DataFrame()
analisis['Magnitud_alta']=mag_high
analisis['Magnitud_baja']=mag_low
analisis['F-Score_Macro']=f_score_macro
analisis['F-Score_Micro']=f_score_micro
analisis['Accuracy']=accuracy
#Show the highest value of F-SCORE
analisis.loc[(analisis['F-Score_Macro']==analisis['F-Score_Macro'].max())]

# Sixth part: join test plots

aprov = pd.concat([random_T_test,random_CC_test, random_control_test])
aprov['Clara_Hech']=aprov['Clara_Hech'].fillna('N')

# Seventh part: Verification
#Put optimal thresholds
a=0.034
b=0.001

#If you want to verified SWIR1 change M7 for M6
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

aprov.loc[(aprov['Clara_Hech']=='N')&(aprov['2018'] =='N')&(aprov['2019'] =='N')&(aprov['2020'] =='N')&(aprov['2021'] =='N'), ['verif']] = ['T']
aprov['pred']='X'
aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='H'), ['pred']] = ['H']
aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='C'), ['pred']] = ['C']
aprov.loc[(aprov['verif'] =='T')&(aprov['Clara_Hech'] =='N'), ['pred']] = ['N']


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

year=['2018','2019','2020','2021']
for h in range(len(year)):
        aprov.loc[(aprov['verif'] =='F')&(aprov['Clara_Hech']=='N')&(aprov[year[h]]=='C')&(aprov['pred']=='X'),['pred']] = ['C']
        aprov.loc[(aprov['verif'] =='F')&(aprov['Clara_Hech']=='N')&(aprov[year[h]]=='H')&(aprov['pred']=='X'),['pred']] = ['H']
y_aprov=np.array(aprov['pred'])
y_true_aprov=np.array(aprov['Clara_Hech'])

# Check the result with a confusion matrix
confusion_matrix(y_true_aprov, y_aprov, labels=['C','H','N'])
