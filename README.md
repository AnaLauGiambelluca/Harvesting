# Harvesting


## English

This folder has all the code implemented in the article called "Identifying forest harvesting practices: clear-cutting and thinning in diverse tree species using dense Landsat time series" written by A. L. Giambelluca, T. Hermosilla, J. Álvarez-Mozos, M. González-Audícana

In order to obtein the research's results it is necessary to open and run the different codes in order:

1- [CCDC's base ](Code/CCDC_Base.txt).
2- [CCDC Raster Conditioning](Code/CCDC_Raster_Conditioning.txt)
3- [Raster to CSV](Code/Raster_to_CSV.py)
4- [Obtain the threshold](Code/Obtein_the_Threshold.py)
5- [Classification and export](Code/Classification_and_export.py)


The result that were analysed in this article is upload in the folder CSV_Results

* Note:
Highlight that in GEE the harvesting plots were used as a single layer. However, the plots are the same as if those stored in Thinning_plot and Clear_cutting_plot were combined, only that the these are already filtered with the final plots used in this work. That is why only the final ones are loaded, in order not to confuse the user of the code. 


## Español

Esta carpeta tiene todo el código implementado en el artículo llamado "Identifying forest harvesting practices: clear-cutting and thinning in diverse tree species using dense Landsat time series" escrito por A. L. Giambelluca, T. Hermosilla, J. Álvarez-Mozos, M. González-Audícana

Para obtener los resultados de la investigación es necesario abrir y ejecutar los diferentes códigos en orden:

1- [CCDC's base ](Code/CCDC_Base.txt).
2- [CCDC Raster Conditioning](Code/CCDC_Raster_Conditioning.txt)
3- [Raster to CSV](Code/Raster_to_CSV.py)
4- [Obtain the threshold](Code/Obtein_the_Threshold.py)
5- [Classification and export](Code/Classification_and_export.py)

Los resultados analizados en este artículo se encuentran disponibles en la carpeta "CSV_Results"

* Nota:
Resaltar que en GEE se utilizaron las parcelas de aprovechamientos como una única capa. No obstante las parcelas son las mismas que si se juntasen las almacenadas en "Thinning_plot" y "Clear_cutting_plot" solo que estas últimas ya se encuentran filtradas con las parcelas definitivas utilizadas en el presente trabajo. Es por eso que solo estan cargadas las finales, a fin de no confundir al usuario del código.
