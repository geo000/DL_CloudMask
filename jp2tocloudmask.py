from keras.models import load_model 
from osgeo import gdal, gdal_array
import sys, subprocess, os, glob, time
import numpy as np
import pandas as pd
import datetime
import win32api

t0 = time.time()
st = datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
print(st)

#points to directory containing jp2 band files
directory_with_jp2s = r''
os.chdir(directory_with_jp2s)

#resamples and converts jp2 band files to 20mX20m pixel images
print('Starting. Resampling and converting jp2s...')
fileList= glob.glob('*B??.jp2')
warp_options = gdal.WarpOptions(format="GTiff",resampleAlg="cubic", width=5490, height=5490,outputType=gdal.GDT_Float32)
for i in fileList:
    img = gdal.Open(i)
    gdal.Warp(str(i[-7:-4]) + '.tif', img, options = warp_options)

#writes out xy coords in gridded XYZ format, to be appended later to predicted values
print('reading xy coords from band 1...')
gdal.Translate(destName='B01.xyz',srcDS='B01.tif',format="XYZ",outputType=gdal.GDT_Byte)
xy = pd.read_csv('B01.xyz',delimiter='\s+',header=None,usecols=[0,1],names=['x','y'])

#scales raw data from 0 to 1, reflectance values
print('calculating new raster values...')
fileList = glob.glob('*B??.tif')
for file in fileList:
    sys.path.append(r'') #point to where your gdal scripts are stored
    gm = os.path.join('C:\\','Users','jbrow','Anaconda3','envs','deeplearning','Scripts','gdal_calc.py') #an example
    merge_command = ["python", gm,"--outfile", file[0:3]+'_updated.tif', '-A' ,file,'--calc','A*0.0001']
    subprocess.call(merge_command,shell=True)

#creates a dataframe to store the new calculated values
df = pd.DataFrame(columns=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])
fileList = glob.glob('*updated.tif')

#fills the dataframe with the new calculated values
print('writing new values to dataframe...')
for file in fileList:
    rasterArray = gdal_array.LoadFile(file)
    write_band = rasterArray.flatten()
    band_number = str(file[0:3])
    df[band_number] = write_band

#loads model
print('loading model...')
model = load_model(r'') #path to the .h5 model file

#dataframe values are predicted on
t3 = time.time()
print('predicting...')
y_pred = model.predict(df.values)
t4 = time.time()
Total = round((t4-t3)/60, 1)
print('Predictions complete, took '+str(Total)+' minutes.')
y_pred_class = np.argmax(y_pred, axis=1)

#predicted values get appended to xy coords from earlier, and then saved
print('creating xyz with predicted values...')
xy['z'] = y_pred_class
xy.to_csv('Mask1.csv',columns = ['x','y','z'],index=False)

#the resulting file is then converted from XYZ/CSV format to tiff using raster info from band 5
print('translating xyz to tif...')
df2 = glob.glob('*B05.jp2')
img = gdal.Open(df2[0])
srs = img.GetProjection()
geo_transform = img.GetGeoTransform()
minx = geo_transform[0]
maxy = geo_transform[3]
maxx = minx + geo_transform[1] * img.RasterXSize
miny = maxy + geo_transform[5] * img.RasterYSize
img=''
gdal.Warp(destNameOrDestDS='_pred.tif',
          srcDSOrSrcDSTab='Mask1.csv',
          width=5490,
          height=5490,
          format="GTiff",
          dstSRS='EPSG:'+srs[-8:-3],
          outputBounds=[minx, miny, maxx, maxy],
          outputType = gdal.GDT_Byte,
         )

#resulting tif has pixel clusters of 8 and smaller sieved out
print('sieving and buffering...')
sys.path.append(r'') #point to where your gdal scripts are stored
gm = os.path.join('C:\\','Users','jbrow','Anaconda3','envs','deeplearning','Scripts','gdal_sieve.py') #an example, change these parameters as needed
sieve_command = ["python", gm,'-st','8','-4','-nomask','-of','GTiff','_pred.tif','_sieved.tif']
subprocess.call(sieve_command,shell=True)

#remaining called out clouds are buffed out by 8 pixels
gm = os.path.join('C:\\','Users','jbrow','Anaconda3','envs','deeplearning','Scripts','gdal_proximity.py')
prox_command = ["python", gm, '-of','GTiff','-distunits','PIXEL','-maxdist','8','-ot','Byte','-fixed-buf-val','1.0','_sieved.tif','_prox.tif']
subprocess.call(prox_command,shell=True)

#the sieved and buffered layers are added together to create the final mask
gm = os.path.join('C:\\','Users','jbrow','Anaconda3','envs','deeplearning','Scripts','gdal_calc.py')
merge_command = ["python", gm,"--outfile", "Final_Cloud_Mask.tif",'-A',"_sieved.tif",'-B','_prox.tif','--calc','A+B']
subprocess.call(merge_command,shell=True)

t1 = time.time()
Total = round((t1-t0)/60, 1)

print('Mask complete!')
print('Algorithm took '+str(Total)+' total minutes to complete')
win32api.MessageBox(0, 'Algorithm took '+str(Total)+' total minutes to complete', 'Mask complete!')
