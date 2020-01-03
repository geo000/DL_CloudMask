from keras.models import load_model
from osgeo import gdal, gdal_array
import sys, subprocess, os, glob, time, datetime
import numpy as np
import pandas as pd

t0 = time.time()
st = datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
print(st)

directory_with_jp2s = r'' #path to jp2s
os.chdir(directory_with_jp2s)

print('Starting. Resampling and converting jp2s...')
fileList= glob.glob('*B??.jp2')
warp_options = gdal.WarpOptions(format="GTiff",resampleAlg="cubic", width=5490, height=5490,outputType=gdal.GDT_Int16)
for i in fileList:
    img = gdal.Open(i)
    gdal.Warp(str(i[-7:-4]) + '.tif', img, options = warp_options)
    
print('reading xy coords from band 1...')
gdal.Translate(destName='B01.xyz',srcDS='B01.tif',format="XYZ",outputType=gdal.GDT_Byte)
xy = pd.read_csv('B01.xyz',delimiter='\s+',header=None,usecols=[0,1],names=['x','y'])

df = pd.DataFrame(columns=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])
fileList= glob.glob('B??.tif')

print('writing reflectance values to dataframe...')
for file in fileList:
    rasterArray = (gdal_array.LoadFile(file)*0.0001)
    write_band = rasterArray.flatten()
    band_number = str(file[0:3])
    df[band_number] = write_band

#split cell here ^
#t0 = time.time()
print('loading model...')
model = load_model(r'') #path to model file

t3 = time.time()
print('predicting...')
y_pred = model.predict(df.values)
t4 = time.time()
Total = round((t4-t3)/60, 1)
print('Predictions complete, took '+str(Total)+' minutes.')
y_pred_class = np.argmax(y_pred, axis=1)

print('creating xyz with predicted values...')
xy['z'] = y_pred_class
xy.to_csv('Mask1.csv',columns = ['x','y','z'],index=False)

print('translating xyz to tif...')
df2 = glob.glob('*B05.jp2')
img = gdal.Open(df2[0])
srs = img.GetProjection()
geo_transform = img.GetGeoTransform()
minx = geo_transform[0]
maxy = geo_transform[3]
maxx = minx + geo_transform[1] * img.RasterXSize
miny = maxy + geo_transform[5] * img.RasterYSize
gdal.Warp(destNameOrDestDS='_pred.tif',
          srcDSOrSrcDSTab='Mask1.csv',
          width=5490,
          height=5490,
          format="GTiff",
          dstSRS='EPSG:'+srs[-8:-3],
          outputBounds=[minx, miny, maxx, maxy],
          outputType = gdal.GDT_Byte)

print('sieving and buffering...')
sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\cpu\Scripts') #path to your gdal scripts
gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','cpu','Scripts','gdal_sieve.py') #change this and subsequent lines to match gdal path
sieve_command = ["python", gm,'-st','8','-4','-nomask','-of','GTiff','_pred.tif','_sieved.tif']
subprocess.call(sieve_command,shell=True)

gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','cpu','Scripts','gdal_proximity.py')
prox_command = ["python", gm, '-of','GTiff','-distunits','PIXEL','-maxdist','8','-ot','Byte','-fixed-buf-val','1.0','_sieved.tif','_prox.tif']
subprocess.call(prox_command,shell=True)

gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','cpu','Scripts','gdal_calc.py')
merge_command = ["python", gm,"--outfile", "Final_Cloud_Mask.tif','-A',"_sieved.tif",'-B','_prox.tif','--calc','A+B']
subprocess.call(merge_command,shell=True)

t1 = time.time()
Total = round((t1-t0)/60, 1)

print('Mask complete!')
print('Algorithm took '+str(Total)+' total minutes to complete')
