#cloud mask from 13 band TIFF
from keras.models import load_model
from osgeo import gdal, gdal_array, osr
import sys, subprocess, os, glob, time, datetime
import numpy as np
import pandas as pd

t0 = time.time()
st = datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
print(st)

out_dir = r'U:\Training_Data\jp2\apabs\Arid\T37RFL_20180102'
allbandsimg = r'T37RFL_S2B_tf20180102.tif'

print('loading image...')
img = gdal_array.LoadFile(os.path.join(out_dir,allbandsimg))
bands = range(0,13)

df = pd.DataFrame(columns=range(0,13))

print('writing reflectance values to dataframe...')
for band in bands:
    rasterArray = (img[band]*0.0001)
    write_band = rasterArray.flatten()
    df[band] = write_band

#split cell here ^
#t0 = time.time()
saved_model = 'BETA_2'
print('loading model...')
model = load_model(r'U:\\Training_Data\\Models\\Model_'+saved_model+'.h5')

t3 = time.time()
print('predicting...')
y_pred = model.predict(df.values)
t4 = time.time()
Total = round((t4-t3)/60, 1)
print('Predictions complete, took '+str(Total)+' minutes.')
y_pred_class = np.argmax(y_pred, axis=1)
reshaped_ypredclass = np.reshape(y_pred_class,(5490,5490))

img = gdal.Open(os.path.join(out_dir,allbandsimg))
srs_prj = img.GetProjection()
geo_transform = img.GetGeoTransform()
driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(os.path.join(out_dir,'_pred.tif'),
                       5490,
                       5490,
                       1,
                       gdal.GDT_Byte)

srs = osr.SpatialReference()
srs.ImportFromEPSG(int(srs_prj[-8:-3]))
dst_ds.SetGeoTransform(list(geo_transform))
dst_ds.SetProjection(srs.ExportToWkt())
dst_band = dst_ds.GetRasterBand(1)
dst_band.SetNoDataValue(0)
dst_band.WriteArray(reshaped_ypredclass)
dst_ds = None

os.chdir(out_dir)
print('sieving and buffering...')
sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\cpu\Scripts')
gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','cpu','Scripts','gdal_sieve.py')
sieve_command = ["python", gm,'-st','8','-4','-nomask','-of','GTiff','_pred.tif','_sieved.tif']
subprocess.call(sieve_command,shell=True)

gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','cpu','Scripts','gdal_proximity.py')
prox_command = ["python", gm, '-of','GTiff','-distunits','PIXEL','-maxdist','8','-ot','Byte','-fixed-buf-val','1.0','_sieved.tif','_prox.tif']
subprocess.call(prox_command,shell=True)

gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','cpu','Scripts','gdal_calc.py')
merge_command = ["python", gm,"--outfile", "Final_Cloud_Mask_"+saved_model+'.tif','-A',"_sieved.tif",'-B','_prox.tif','--calc','A+B']
subprocess.call(merge_command,shell=True)

t1 = time.time()
Total = round((t1-t0)/60, 1)

print('Mask complete!')
print('Algorithm took '+str(Total)+' total minutes to complete')
