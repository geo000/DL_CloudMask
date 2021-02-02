import os, random, glob, sys, subprocess, time, cv2, datetime, shutil, contextlib
import numpy as np
from tqdm import notebook
from itertools import chain
from skimage.io import imread
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tifffile
from osgeo import gdal, gdal_array, osr

#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
#config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

t0 = time.time()

st = datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
print(st)
print("Starting, creating 13 band stack...")

directory_with_jp2s = r'D:\NewCloudTrials\Test\T21HUB_20200905'
gdal_path = r'C:\Users\jbrow\anaconda3\envs\test\Scripts'
os.chdir(directory_with_jp2s)
try:
    os.mkdir('Result')
except:
    pass
try:
    os.mkdir('Tiles')
except:
    pass
file_name = directory_with_jp2s[-15:]
granule_tag = directory_with_jp2s[-15:-9]

band = glob.glob("*B??.jp2")

gdal_merge_path = os.path.join(gdal_path,'gdal_merge.py')
separate = '-separate'
pixel_size = '20'
outputFile = file_name+"_13band_20m.tif"

gdal_merge_str = 'python {0} {1} -ps {2} {3} -o {4} {5} {6} {7} {8} {9} {10} {11} {12} {13} {14} {15} {16} {17}'
gdal_merge_process = gdal_merge_str.format(
gdal_merge_path,
separate,
pixel_size,
pixel_size,
outputFile,
band[0],band[1],band[2],band[3],band[4],band[5],band[6],band[7],band[12],band[8],band[9],band[10],band[11])
os.system(gdal_merge_process)

width = 5490
height = 5490
tilesize = 305

print('Tiling images...')
for i in range(0, width, tilesize):
    for j in range(0, height, tilesize):
        gdal.Translate(destName = 'Tiles/'+str(i)+'_'+str(j)+'_'+granule_tag+'.png',
                      srcDS = file_name+'_13band_20m.tif',
                      srcWin = [i,j,tilesize,tilesize],
                      bandList = [1],
                      noData = None,
                      format = "PNG"
                      )
        
for i in range(0, width, tilesize):
    for j in range(0, height, tilesize):
        gdal.Translate(destName = 'Tiles/'+str(i)+'_'+str(j)+'_'+granule_tag+'.tif',
                      srcDS = file_name+'_13band_20m.tif',
                      srcWin = [i,j,tilesize,tilesize],
                      noData = None,
                      format = "GTiff"
                      )
    
im_height = 256
im_width = 256
os.chdir('Tiles/')

png_list = glob.glob('*.png')
for i in png_list:
    os.remove(i)

ids = glob.glob('*.tif')
for i in ids:
    if '.xml' in i:
        ids.remove(i)

try:
    ids.remove('Thumbs.db')
except:
    pass

print("No. of images = ", len(ids))

X_test = np.zeros((len(ids), im_height, im_width, 13), dtype=np.float32)

print('Loading tiles for prediction...')

for n, id_ in notebook.tqdm(enumerate(ids), total=len(ids)):
    img = (tifffile.imread(id_))*0.0001
    x_img = resize(img, (im_height, im_width, 13), mode = 'edge', preserve_range = True)
    X_test[n] = np.round_(x_img,4)

print('Loading model...')
model = load_model(r"D:\NewCloudTrials\Models\Temperate_Model_108.h5")

print('Predicting...')
t3 = time.time()
result = model.predict(X_test,batch_size=1)
t4 = time.time()
Total = round((t4-t3)/60, 1)
print('Predictions complete, took '+str(Total)+' minutes.')

os.chdir('..')
os.chdir('Result/')

print('Saving results...')
for i,k in enumerate(ids):
    x_img = cv2.resize(result[i].astype('float32'), (305, 305))
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(k[0:-4]+'.png',305,305,1,gdal.GDT_Float32)
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.WriteArray(x_img)
    dst_ds = None
    #cv2.imwrite(k[0:-4]+'.png',x_img)

os.chdir('..')
os.chdir('Tiles/')
metadata_list = glob.glob('*.xml')
os.chdir('..')
dest1 = 'Result/'
source = 'Tiles/'

print('Copying over metadata...')
for f in metadata_list:
    shutil.copy2(source+f, dest1)

os.chdir('Result/')
mosaic_list = glob.glob('*.png')

print('Mosaicking results...')
gdal.Warp(destNameOrDestDS='_pred.tif',
          srcDSOrSrcDSTab=mosaic_list,
          width=5490,
          height=5490,
          format='GTiff',
          outputType=gdal.GDT_Float32
          )

gdal_calc_path = os.path.join(gdal_path, 'gdal_calc.py')
input_file_path = '_pred.tif'
output_file_path = '_pred_filtered.tif'
calc_expr = '"A>=0.95"' #Threshold number for cloud detection 0-0.99, 0.99 is the most thresholded
typeof = '"Byte"'
gdal_calc_str = 'python {0} -A {1} --outfile={2} --calc={3} --type={4}'
gdal_calc_process = gdal_calc_str.format(gdal_calc_path, input_file_path, output_file_path, calc_expr, typeof)
os.system(gdal_calc_process)

#Sieve
gdal_sieve_path = os.path.join(gdal_path,'gdal_sieve.py')
input_file_path = '_pred_filtered.tif'
output_file_path = '_sieved.tif'
threshold = '10' #Clusters of 10 pixels and under will be removed
connectedness = '-8' #Pixel cluster alg will consider pixels on the 4 corners and 4 edges of each pixel
outputFormat = 'GTiff'
nomask = '-nomask'

gdal_sieve_str = 'python {0} -st {1} {2} {3} -of {4} {5} {6}'
gdal_sieve_process = gdal_sieve_str.format(gdal_sieve_path, threshold, connectedness, nomask, outputFormat, input_file_path, output_file_path)
os.system(gdal_sieve_process)

#Proximity
gdal_prox_path = os.path.join(gdal_path,'gdal_proximity.py')
input_file_path = '_sieved.tif'
output_file_path = '_prox.tif'
dist_units = 'PIXEL' #The buffer applied will be measured by pixels, as opposed to meters
max_dist = '3' #Cloud buffer amount.  Note that these are 20m pixels, so a 3 pixel buffer is a 6 pixel 10m buffer.
fixed_buffer_val = '1.0'

gdal_prox_str = 'python {0} -of {1} -distunits {2} -maxdist {3} -ot {4} -fixed-buf-val {5} {6} {7}'
gdal_prox_process = gdal_prox_str.format(gdal_prox_path,outputFormat,dist_units,max_dist,typeof,fixed_buffer_val,input_file_path,output_file_path)
os.system(gdal_prox_process)

#Add proximity and sieve, creating final mask
out_mask = file_name+'_cloudmask_delete.tif'
out_mask2 = file_name+'_cloudmask.tif'
A = '_prox.tif'
B = '_sieved.tif'
calc_expr = '"A+B"' #Buffer layer created by gdal_proximity is produced without original data, so it must be combined with gdal_sieve output
creation_option = 'COMPRESSED=YES'
outputFormat = 'GTiff'

gdal_calc_str = 'python {0} --calc {1} -A {2} -B {3} --creation-option {4} --format {5} --outfile {6}'
gdal_calc_process = gdal_calc_str.format(gdal_calc_path,calc_expr,A,B,creation_option,outputFormat,out_mask)
os.system(gdal_calc_process)

img = gdal.Open(out_mask)
srs_prj = img.GetProjection()
geo_transform = img.GetGeoTransform()
cloud_array = img.GetRasterBand(1).ReadAsArray()
cloud_array[cloud_array == 255] = 0

driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create(out_mask2,5490,5490,1,gdal.GDT_Byte,options=["COMPRESS=LZW"])
srs = osr.SpatialReference()
srs.ImportFromEPSG(int(srs_prj[-8:-3]))
dst_ds.SetGeoTransform(list(geo_transform))
dst_ds.SetProjection(srs.ExportToWkt())
dst_band = dst_ds.GetRasterBand(1)
dst_band.WriteArray(cloud_array)
dst_ds = None

os.chdir('..')

t1 = time.time()
Total = round((t1-t0)/60, 1)

os.chdir('..')
print('Mask complete!')
print('Algorithm took '+str(Total)+' total minutes to complete')
