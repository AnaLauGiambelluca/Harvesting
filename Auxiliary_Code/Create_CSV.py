#Code to create a CSV with the mean magnitude of each plot.

# Import libraries

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
import math
import statistics

# First part: upload vectorial and raster layers

# Vectorial layers
direct='Path'
os.chdir(direct)
SHP_T=gpd.read_file('Thinning_plot.shp')
SHP_C=gpd.read_file('Control_plot.shp')
SHP_CC=gpd.read_file('Clear_cutting_plot.shp')

# Raster layers
direct_raster='Path_raster'
os.chdir(direct_raster)
# Logging plots
B_t1=gdal.Open('CCDC_A_t1_prob_B7_B6.tif')
B_t2=gdal.Open('CCDC_A_t2_prob_B7_B6.tif')
B_t3=gdal.Open('CCDC_A_t3_prob_B7_B6.tif')
B_t4=gdal.Open('CCDC_A_t4_prob_B7_B6.tif')
B_t5=gdal.Open('CCDC_A_t5_prob_B7_B6.tif')
B_t6=gdal.Open('CCDC_A_t6_prob_B7_B6.tif')

#Control plot. To assessed this uncomment and comment the previous code 
# B_t1=gdal.Open('CCDC_C_t1_prob_B7_B6.tif')
# B_t2=gdal.Open('CCDC_C_t2_prob_B7_B6.tif')
# B_t3=gdal.Open('CCDC_C_t3_prob_B7_B6.tif')
# B_t4=gdal.Open('CCDC_C_t4_prob_B7_B6.tif')
# B_t5=gdal.Open('CCDC_C_t5_prob_B7_B6.tif')
# B_t6=gdal.Open('CCDC_C_t6_prob_B7_B6.tif')


# Second part: Get all bands


#Date

B1_t1=B_t1.GetRasterBand(1).ReadAsArray().astype(np.float32)
B1_t2=B_t2.GetRasterBand(1).ReadAsArray().astype(np.float32)
B1_t3=B_t3.GetRasterBand(1).ReadAsArray().astype(np.float32)
B1_t4=B_t4.GetRasterBand(1).ReadAsArray().astype(np.float32)
B1_t5=B_t5.GetRasterBand(1).ReadAsArray().astype(np.float32)
B1_t6=B_t6.GetRasterBand(1).ReadAsArray().astype(np.float32)

#Magnitude SWIR2 band
B2_t1=B_t1.GetRasterBand(2).ReadAsArray().astype(np.float32)
B2_t2=B_t2.GetRasterBand(2).ReadAsArray().astype(np.float32)
B2_t3=B_t3.GetRasterBand(2).ReadAsArray().astype(np.float32)
B2_t4=B_t4.GetRasterBand(2).ReadAsArray().astype(np.float32)
B2_t5=B_t5.GetRasterBand(2).ReadAsArray().astype(np.float32)
B2_t6=B_t6.GetRasterBand(2).ReadAsArray().astype(np.float32)

# Magnitude SWIR1 band
B3_t1=B_t1.GetRasterBand(3).ReadAsArray().astype(np.float32)
B3_t2=B_t2.GetRasterBand(3).ReadAsArray().astype(np.float32)
B3_t3=B_t3.GetRasterBand(3).ReadAsArray().astype(np.float32)
B3_t4=B_t4.GetRasterBand(3).ReadAsArray().astype(np.float32)
B3_t5=B_t5.GetRasterBand(3).ReadAsArray().astype(np.float32)
B3_t6=B_t6.GetRasterBand(3).ReadAsArray().astype(np.float32)

# Probability of change 
b6_t1=B_t1.GetRasterBand(4).ReadAsArray().astype(np.float32)
b6_t2=B_t2.GetRasterBand(4).ReadAsArray().astype(np.float32)
b6_t3=B_t3.GetRasterBand(4).ReadAsArray().astype(np.float32)
b6_t4=B_t4.GetRasterBand(4).ReadAsArray().astype(np.float32)
b6_t5=B_t5.GetRasterBand(4).ReadAsArray().astype(np.float32)
b6_t6=B_t6.GetRasterBand(4).ReadAsArray().astype(np.float32)


# Third part: Create arrays. One for date, one for probability and one for both maginitude (SWIR1 and SWIR2). This will be done for each year

capa_1 = np.where((B1_t1 < 2016) & (B1_t1 >=2015) , B1_t1, 0)
capa_2 = np.where((B1_t2 < 2016) & (B1_t2 >=2015) , B1_t2, 0)
capa_3 = np.where((B1_t3 < 2016) & (B1_t3 >=2015) , B1_t3, 0)
capa_4 = np.where((B1_t4 < 2016) & (B1_t4 >=2015) , B1_t4, 0)
capa_5 = np.where((B1_t5 < 2016) & (B1_t5 >=2015) , B1_t5, 0)
capa_6 = np.where((B1_t6 < 2016) & (B1_t6 >=2015) , B1_t6, 0)

# Date of 2015
t_2015=capa_1+capa_2+capa_3+capa_4+capa_5+capa_6

Prob1 = np.where((capa_1 !=0) , b6_t1, 0)
Prob2 = np.where((capa_2 !=0) , b6_t2, 0)
Prob3 = np.where((capa_3 !=0) , b6_t3, 0)
Prob6 = np.where((capa_4 !=0) , b6_t4, 0)
Prob5 = np.where((capa_5 !=0) , b6_t5, 0)
Prob6 = np.where((capa_6 !=0) , b6_t6, 0)

#Probability of 2015
Prob_2015=Prob1+Prob2+Prob3+Prob6+Prob5+Prob6

mag_B7_1=np.where((Prob1 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob2 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob3 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob6 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob5 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob6 ==1) , B2_t6, 0)

# SWIR2 magnitude of 2015
Mag_B7_2015=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6


mag_b6_1=np.where((Prob1 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob2 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob3 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob6 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob5 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob6 ==1) , B3_t6, 0)

#SWIR1 maginitude of 2015
Mag_b6_2015=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6

# The same for the following years

#2016
t_1 = np.where((B1_t1 < 2017) & (B1_t1 >=2016) , B1_t1, 0)
t_2 = np.where((B1_t2 < 2017) & (B1_t2 >=2016) , B1_t2, 0)
t_3 = np.where((B1_t3 < 2017) & (B1_t3 >=2016) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2017) & (B1_t4 >=2016) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2017) & (B1_t5 >=2016) , B1_t5, 0)
t_6 = np.where((B1_t6 < 2017) & (B1_t6 >=2016) , B1_t6, 0)
t_2016=t_1+t_2+t_3+t_4+t_5+t_6
Prob_1_2016 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2016 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2016 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2016 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2016 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2016 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2016=Prob_1_2016+Prob_2_2016+Prob_3_2016+Prob_4_2016+Prob_5_2016+Prob_6_2016
mag_B7_1=np.where((Prob_1_2016 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2016 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2016 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2016 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2016 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2016 ==1) , B2_t6, 0)
Mag_B7_2016=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2016 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2016 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2016 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2016 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2016 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2016 ==1) , B3_t6, 0)
Mag_b6_2016=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2016,Prob_2_2016,Prob_3_2016,Prob_4_2016,Prob_5_2016,Prob_6_2016
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
#2017
t_1 = np.where((B1_t1 < 2018) & (B1_t1 >=2017) , B1_t1, 0)
t_2= np.where((B1_t2 < 2018) & (B1_t2 >=2017) , B1_t2,0)
t_3 = np.where((B1_t3 < 2018) & (B1_t3 >=2017) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2018) & (B1_t4 >=2017) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2018) & (B1_t5 >=2017) , B1_t5,0)
t_6= np.where((B1_t6 < 2018) & (B1_t6 >=2017) , B1_t6, 0)
t_2017=t_1+t_2+t_3+t_4+t_5+t_6
#Para 2017 3
Prob_1_2017 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2017 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2017 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2017 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2017 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2017 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2017=Prob_1_2017+Prob_2_2017+Prob_3_2017+Prob_4_2017+Prob_5_2017+Prob_6_2017
mag_B7_1=np.where((Prob_1_2017 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2017 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2017 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2017 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2017 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2017 ==1) , B2_t6, 0)
Mag_B7_2017=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2017 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2017 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2017 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2017 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2017 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2017 ==1) , B3_t6, 0)
Mag_b6_2017=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del  t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2017,Prob_2_2017,Prob_3_2017,Prob_4_2017,Prob_5_2017,Prob_6_2017
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
#2018
t_1= np.where((B1_t1 < 2019) & (B1_t1 >=2018) , B1_t1, 0)
t_2 = np.where((B1_t2 < 2019) & (B1_t2 >=2018) , B1_t2, 0)
t_3 = np.where((B1_t3 < 2019) & (B1_t3 >=2018) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2019) & (B1_t4 >=2018) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2019) & (B1_t5 >=2018) , B1_t5, 0)
t_6= np.where((B1_t6 < 2019) & (B1_t6 >=2018) , B1_t6, 0)
t_2018=t_1+t_2+t_3+t_4+t_5+t_6
Prob_1_2018 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2018 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2018 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2018 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2018 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2018 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2018=Prob_1_2018+Prob_2_2018+Prob_3_2018+Prob_4_2018+Prob_5_2018+Prob_6_2018
mag_B7_1=np.where((Prob_1_2018 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2018 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2018 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2018 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2018 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2018 ==1) , B2_t6, 0)
Mag_B7_2018=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2018 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2018 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2018 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2018 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2018 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2018 ==1) , B3_t6, 0)
Mag_b6_2018=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del  t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2018,Prob_2_2018,Prob_3_2018,Prob_4_2018,Prob_5_2018,Prob_6_2018
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
#2019
t_1 = np.where((B1_t1 < 2020) & (B1_t1 >=2019) , B1_t1, 0)
t_2 = np.where((B1_t2 < 2020) & (B1_t2 >=2019) , B1_t2, 0)
t_3 = np.where((B1_t3 < 2020) & (B1_t3 >=2019) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2020) & (B1_t4 >=2019) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2020) & (B1_t5 >=2019) , B1_t5, 0)
t_6 = np.where((B1_t6 < 2020) & (B1_t6 >=2019) , B1_t6, 0)
t_2019=t_1+t_2+t_3+t_4+t_5+t_6
Prob_1_2019 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2019 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2019 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2019 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2019 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2019 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2019=Prob_1_2019+Prob_2_2019+Prob_3_2019+Prob_4_2019+Prob_5_2019+Prob_6_2019
mag_B7_1=np.where((Prob_1_2019 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2019 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2019 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2019 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2019 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2019 ==1) , B2_t6, 0)
Mag_B7_2019=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2019 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2019 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2019 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2019 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2019 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2019 ==1) , B3_t6, 0)
Mag_b6_2019=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del  t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2019,Prob_2_2019,Prob_3_2019,Prob_4_2019,Prob_5_2019,Prob_6_2019
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
#2020
t_1 = np.where((B1_t1 < 2021) & (B1_t1 >=2020) , B1_t1, 0)
t_2 = np.where((B1_t2 < 2021) & (B1_t2 >=2020) , B1_t2, 0)
t_3 = np.where((B1_t3 < 2021) & (B1_t3 >=2020) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2021) & (B1_t4 >=2020) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2021) & (B1_t5 >=2020) , B1_t5, 0)
t_6= np.where((B1_t6 < 2021) & (B1_t6 >=2020) , B1_t6, 0)
t_2020=t_1+t_2+t_3+t_4+t_5+t_6
#Para 2020 4
Prob_1_2020 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2020 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2020 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2020 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2020 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2020 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2020=Prob_1_2020+Prob_2_2020+Prob_3_2020+Prob_4_2020+Prob_5_2020+Prob_6_2020
mag_B7_1=np.where((Prob_1_2020 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2020 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2020 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2020 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2020 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2020 ==1) , B2_t6, 0)
Mag_B7_2020=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2020 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2020 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2020 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2020 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2020 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2020 ==1) , B3_t6, 0)
Mag_b6_2020=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del  t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2020,Prob_2_2020,Prob_3_2020,Prob_4_2020,Prob_5_2020,Prob_6_2020
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
#2021
t_1 = np.where((B1_t1 < 2022) & (B1_t1 >=2021) , B1_t1, 0)
t_2 = np.where((B1_t2 < 2022) & (B1_t2 >=2021) , B1_t2, 0)
t_3 = np.where((B1_t3 < 2022) & (B1_t3 >=2021) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2022) & (B1_t4 >=2021) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2022) & (B1_t5 >=2021) , B1_t5, 0)
t_6 = np.where((B1_t6 < 2022) & (B1_t6 >=2021) , B1_t6, 0)
t_2021=t_1+t_2+t_3+t_4+t_5+t_6
Prob_1_2021 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2021 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2021 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2021 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2021 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2021 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2021=Prob_1_2021+Prob_2_2021+Prob_3_2021+Prob_4_2021+Prob_5_2021+Prob_6_2021
mag_B7_1=np.where((Prob_1_2021 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2021 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2021 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2021 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2021 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2021 ==1) , B2_t6, 0)
Mag_B7_2021=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2021 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2021 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2021 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2021 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2021 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2021 ==1) , B3_t6, 0)
Mag_b6_2021=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del  t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2021,Prob_2_2021,Prob_3_2021,Prob_4_2021,Prob_5_2021,Prob_6_2021
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
#2022
t_1 = np.where((B1_t1 < 2023) & (B1_t1 >=2022) , B1_t1, 0)
t_2 = np.where((B1_t2 < 2023) & (B1_t2 >=2022) , B1_t2, 0)
t_3 = np.where((B1_t3 < 2023) & (B1_t3 >=2022) , B1_t3, 0)
t_4 = np.where((B1_t4 < 2023) & (B1_t4 >=2022) , B1_t4, 0)
t_5 = np.where((B1_t5 < 2023) & (B1_t5 >=2022) , B1_t5, 0)
t_6 = np.where((B1_t6 < 2023) & (B1_t6 >=2022) , B1_t6, 0)
t_2022=t_1+t_2+t_3+t_4+t_5+t_6
Prob_1_2022 = np.where((t_1 !=0) , b6_t1, 0)
Prob_2_2022 = np.where((t_2 !=0) , b6_t2, 0)
Prob_3_2022 = np.where((t_3 !=0) , b6_t3, 0)
Prob_4_2022 = np.where((t_4 !=0) , b6_t4, 0)
Prob_5_2022 = np.where((t_5 !=0) , b6_t5, 0)
Prob_6_2022 = np.where((t_6 !=0) , b6_t6, 0)
Prob_2022=Prob_1_2022+Prob_2_2022+Prob_3_2022+Prob_4_2022+Prob_5_2022+Prob_6_2022
mag_B7_1=np.where((Prob_1_2022 ==1) , B2_t1, 0)
mag_B7_2=np.where((Prob_2_2022 ==1) , B2_t2, 0)
mag_B7_3=np.where((Prob_3_2022 ==1) , B2_t3, 0)
mag_B7_4=np.where((Prob_4_2022 ==1) , B2_t4, 0)
mag_B7_5=np.where((Prob_5_2022 ==1) , B2_t5, 0)
mag_B7_6=np.where((Prob_6_2022 ==1) , B2_t6, 0)
Mag_B7_2022=mag_B7_1+mag_B7_2+mag_B7_3+mag_B7_4+mag_B7_5+mag_B7_6
mag_b6_1=np.where((Prob_1_2022 ==1) , B3_t1, 0)
mag_b6_2=np.where((Prob_2_2022 ==1) , B3_t2, 0)
mag_b6_3=np.where((Prob_3_2022 ==1) , B3_t3, 0)
mag_b6_4=np.where((Prob_4_2022 ==1) , B3_t4, 0)
mag_b6_5=np.where((Prob_5_2022 ==1) , B3_t5, 0)
mag_b6_6=np.where((Prob_6_2022 ==1) , B3_t6, 0)
Mag_b6_2022=mag_b6_1+mag_b6_2+mag_b6_3+mag_b6_4+mag_b6_5+mag_b6_6
del  t_1,t_2,t_3,t_4,t_5,t_6
del Prob_1_2022,Prob_2_2022,Prob_3_2022,Prob_4_2022,Prob_5_2022,Prob_6_2022
del mag_B7_1,mag_B7_2,mag_B7_3,mag_B7_4,mag_B7_5,mag_B7_6
del mag_b6_1,mag_b6_2,mag_b6_3,mag_b6_4,mag_b6_5,mag_b6_6
t_2016=np.floor(t_2016)
t_2017=np.floor(t_2017)
t_2018=np.floor(t_2018)
t_2019=np.floor(t_2019)
t_2020=np.floor(t_2020)
t_2021=np.floor(t_2021)
t_2022=np.floor(t_2022)


# Fourth part: upload raster of plots with ID

#For each type of plot you have to uncomment the plot selected and comment the others. In this example the analysis  will be in thinning plots
Direct_ID='Path'
os.chdir(Direct_ID)
#cortas=gdal.Open('Raster_Control.tiff')
cortas=gdal.Open('Raster_CC.tiff')
#cortas=gdal.Open('Raster_T.tiff')

#Get the ID band
cortas=cortas.GetRasterBand(1).ReadAsArray().astype(np.float32)
#Mask only the plot
cortas= np.where(cortas !=0 , cortas, -1)
# Create a list with all ID
n_parcelas=list(set(cortas[cortas!=-1]))
# Create a 1D array
cortas=cortas.flatten()

# Create list where information will be saved.
val_2016=[]
val_2017=[]
val_2018=[]
val_2019=[]
val_2020=[]
val_2021=[]
val_2022=[]
zeros=[]
conteo=[]
nombre=[]  
P1_16=[]
P1_17=[]
P1_18=[]
P1_19=[]
P1_20=[]
P1_21=[]
P1_22=[]
M7_16=[]
M7_17=[]
M7_18=[]
M7_19=[]
M7_20=[]
M7_21=[]
M7_22=[]
Mb6_16=[]
Mb6_17=[]
Mb6_18=[]
Mb6_19=[]
Mb6_20=[]
Mb6_21=[]
Mb6_22=[]

porcent=0

dC = pd.DataFrame()
#Loop where for each date, the number of pixels with a change, the number of pixels with a probable change and the average magnitude of change of SWIR2 and SWIR1 will be stored. 
for y in range(len(n_parcelas)):
    print((porcent/len(n_parcelas))*100)
    parcelas=np.where((cortas ==n_parcelas[y]) , 1, -1)

    c_2016=t_2016.copy()
    c_2017=t_2017.copy()
    c_2018=t_2018.copy()
    c_2019=t_2019.copy()
    c_2020=t_2020.copy()
    c_2021=t_2021.copy()
    c_2022=t_2022.copy()    
    c_2016=c_2016.flatten()
    c_2017=c_2017.flatten()
    c_2018=c_2018.flatten()
    c_2019=c_2019.flatten()
    c_2020=c_2020.flatten()
    c_2021=c_2021.flatten()
    c_2022=c_2022.flatten()
    P1_2016=Prob_2016.copy()
    P1_2017=Prob_2017.copy()
    P1_2018=Prob_2018.copy()
    P1_2019=Prob_2019.copy()
    P1_2020=Prob_2020.copy()
    P1_2021=Prob_2021.copy()
    P1_2022=Prob_2022.copy()    
    P1_2016=P1_2016.flatten()
    P1_2017=P1_2017.flatten()
    P1_2018=P1_2018.flatten()
    P1_2019=P1_2019.flatten()
    P1_2020=P1_2020.flatten()
    P1_2021=P1_2021.flatten()
    P1_2022=P1_2022.flatten()
    M7_2016=Mag_B7_2016.copy()
    M7_2017=Mag_B7_2017.copy()
    M7_2018=Mag_B7_2018.copy()
    M7_2019=Mag_B7_2019.copy()
    M7_2020=Mag_B7_2020.copy()
    M7_2021=Mag_B7_2021.copy()
    M7_2022=Mag_B7_2022.copy()
    M7_2016=M7_2016.flatten()
    M7_2017=M7_2017.flatten()
    M7_2018=M7_2018.flatten()
    M7_2019=M7_2019.flatten()
    M7_2020=M7_2020.flatten()
    M7_2021=M7_2021.flatten()
    M7_2022=M7_2022.flatten()
    Mb6_2016=Mag_b6_2016.copy()
    Mb6_2017=Mag_b6_2017.copy()
    Mb6_2018=Mag_b6_2018.copy()
    Mb6_2019=Mag_b6_2019.copy()
    Mb6_2020=Mag_b6_2020.copy()
    Mb6_2021=Mag_b6_2021.copy()
    Mb6_2022=Mag_b6_2022.copy()
    Mb6_2016=Mb6_2016.flatten()
    Mb6_2017=Mb6_2017.flatten()
    Mb6_2018=Mb6_2018.flatten()
    Mb6_2019=Mb6_2019.flatten()
    Mb6_2020=Mb6_2020.flatten()
    Mb6_2021=Mb6_2021.flatten()
    Mb6_2022=Mb6_2022.flatten()
    c_2016= np.where((cortas ==n_parcelas[y]),c_2016, -1)
    c_2017= np.where((cortas ==n_parcelas[y]),c_2017, -1)
    c_2018= np.where((cortas ==n_parcelas[y]),c_2018, -1)
    c_2019= np.where((cortas ==n_parcelas[y]),c_2019, -1)
    c_2020= np.where((cortas ==n_parcelas[y]),c_2020, -1)
    c_2021= np.where((cortas ==n_parcelas[y]),c_2021, -1)
    c_2022= np.where((cortas ==n_parcelas[y]),c_2022, -1)
    P1_2016=np.where((cortas ==n_parcelas[y]),P1_2016, 999)
    P1_2017=np.where((cortas ==n_parcelas[y]),P1_2017, 999)
    P1_2018=np.where((cortas ==n_parcelas[y]),P1_2018, 999)
    P1_2019=np.where((cortas ==n_parcelas[y]),P1_2019, 999)
    P1_2020=np.where((cortas ==n_parcelas[y]),P1_2020, 999)
    P1_2021=np.where((cortas ==n_parcelas[y]),P1_2021, 999)
    P1_2022=np.where((cortas ==n_parcelas[y]),P1_2022, 999)
    c_2016=list(c_2016[c_2016 != -1])
    c_2017=list(c_2017[c_2017 != -1])
    c_2018=list(c_2018[c_2018 != -1])
    c_2019=list(c_2019[c_2019 != -1])
    c_2020=list(c_2020[c_2020 != -1])
    c_2021=list(c_2021[c_2021 != -1])
    c_2022=list(c_2022[c_2022 != -1])
    P1_2016=list(P1_2016[P1_2016 != 999])
    P1_2017=list(P1_2017[P1_2017 != 999])
    P1_2018=list(P1_2018[P1_2018 != 999])
    P1_2019=list(P1_2019[P1_2019 != 999])
    P1_2020=list(P1_2020[P1_2020 != 999])
    P1_2021=list(P1_2021[P1_2021 != 999])
    P1_2022=list(P1_2022[P1_2022 != 999])
    val_2016.append(c_2016.count(2016))
    val_2017.append(c_2017.count(2017))
    val_2018.append(c_2018.count(2018))
    val_2019.append(c_2019.count(2019))
    val_2020.append(c_2020.count(2020))
    val_2021.append(c_2021.count(2021))
    val_2022.append(c_2022.count(2022))                    
    P1_16.append(P1_2016.count(1))
    P1_17.append(P1_2017.count(1))
    P1_18.append(P1_2018.count(1))
    P1_19.append(P1_2019.count(1))
    P1_20.append(P1_2020.count(1))
    P1_21.append(P1_2021.count(1))
    P1_22.append(P1_2022.count(1))
    M7_2016= np.where((cortas ==n_parcelas[y]),M7_2016, -1)
    M7_2017= np.where((cortas ==n_parcelas[y]),M7_2017, -1)
    M7_2018= np.where((cortas ==n_parcelas[y]),M7_2018, -1)
    M7_2019= np.where((cortas ==n_parcelas[y]),M7_2019, -1)
    M7_2020= np.where((cortas ==n_parcelas[y]),M7_2020, -1)
    M7_2021= np.where((cortas ==n_parcelas[y]),M7_2021, -1)
    M7_2022= np.where((cortas ==n_parcelas[y]),M7_2022, -1)
    M7_2016=list(M7_2016[M7_2016 != -1])
    M7_2017=list(M7_2017[M7_2017 != -1])
    M7_2018=list(M7_2018[M7_2018 != -1])
    M7_2019=list(M7_2019[M7_2019 != -1])
    M7_2020=list(M7_2020[M7_2020 != -1])
    M7_2021=list(M7_2021[M7_2021 != -1])
    M7_2022=list(M7_2022[M7_2022 != -1])
    M7_16.append(statistics.mean(M7_2016))
    M7_17.append(statistics.mean(M7_2017))
    M7_18.append(statistics.mean(M7_2018))
    M7_19.append(statistics.mean(M7_2019))
    M7_20.append(statistics.mean(M7_2020))
    M7_21.append(statistics.mean(M7_2021))
    M7_22.append(statistics.mean(M7_2022))         
    Mb6_2016= np.where((cortas ==n_parcelas[y]),Mb6_2016, -1)
    Mb6_2017= np.where((cortas ==n_parcelas[y]),Mb6_2017, -1)
    Mb6_2018= np.where((cortas ==n_parcelas[y]),Mb6_2018, -1)
    Mb6_2019= np.where((cortas ==n_parcelas[y]),Mb6_2019, -1)
    Mb6_2020= np.where((cortas ==n_parcelas[y]),Mb6_2020, -1)
    Mb6_2021= np.where((cortas ==n_parcelas[y]),Mb6_2021, -1)
    Mb6_2022= np.where((cortas ==n_parcelas[y]),Mb6_2022, -1)
    Mb6_2016_v=list(Mb6_2016[Mb6_2016 != -1])
    Mb6_2017_v=list(Mb6_2017[Mb6_2017 != -1])
    Mb6_2018_v=list(Mb6_2018[Mb6_2018 != -1])
    Mb6_2019_v=list(Mb6_2019[Mb6_2019 != -1])
    Mb6_2020_v=list(Mb6_2020[Mb6_2020 != -1])
    Mb6_2021_v=list(Mb6_2021[Mb6_2021 != -1])
    Mb6_2022_v=list(Mb6_2022[Mb6_2022 != -1])
    Mb6_16.append(statistics.mean(Mb6_2016_v))
    Mb6_17.append(statistics.mean(Mb6_2017_v))
    Mb6_18.append(statistics.mean(Mb6_2018_v))
    Mb6_19.append(statistics.mean(Mb6_2019_v))
    Mb6_20.append(statistics.mean(Mb6_2020_v))
    Mb6_21.append(statistics.mean(Mb6_2021_v))
    Mb6_22.append(statistics.mean(Mb6_2022_v))         
    parcelas=list(parcelas[parcelas != -1])
    conteo.append(parcelas.count(1))
    nombre.append(n_parcelas[y])
    porcent+=1

#Save the information into a DataFrame
dC['ID']=nombre
dC['N_pixel']=conteo
# Amount of pixels that changed in each year
dC['N_pixel_16']=val_2016
dC['N_pixel_17']=val_2017
dC['N_pixel_18']=val_2018
dC['N_pixel_19']=val_2019
dC['N_pixel_20']=val_2020
dC['N_pixel_21']=val_2021
dC['N_pixel_22']=val_2022
dC['P1_16']=P1_16
dC['P1_17']=P1_17
dC['P1_18']=P1_18
dC['P1_19']=P1_19
dC['P1_20']=P1_20
dC['P1_21']=P1_21
dC['P1_22']=P1_22
dC['M7_16']=M7_16
dC['M7_17']=M7_17
dC['M7_18']=M7_18
dC['M7_19']=M7_19
dC['M7_20']=M7_20
dC['M7_21']=M7_21
dC['M7_22']=M7_22
dC['Mb6_16']=Mb6_16
dC['Mb6_17']=Mb6_17
dC['Mb6_18']=Mb6_18
dC['Mb6_19']=Mb6_19
dC['Mb6_20']=Mb6_20
dC['Mb6_21']=Mb6_21
dC['Mb6_22']=Mb6_22

#Export the CSV with all
H=dC.copy()
H.to_csv('Thinning_B7_B6.csv', header=True, index=True)
