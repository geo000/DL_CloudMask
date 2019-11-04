import os, random, glob, sys, subprocess, time, cv2, datetime, shutil, contextlib
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")
%matplotlib inline

from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread, imshow, concatenate_images, imsave
from skimage.transform import resize
from skimage.morphology import label
from sklearn.model_selection import train_test_split
from osgeo import gdal, gdal_array

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

t0 = time.time()

st = datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
print(st)
print("Starting, creating 4 band stack...")

directory_with_jp2s = r''
os.chdir(directory_with_jp2s)
try:
    os.mkdir('Result')
except:
    pass
try:
    os.mkdir('Tiles')
except:
    pass
file_name = directory_with_jp2s[-15:] #This will likely need to be changed
granule_tag = directory_with_jp2s[-15:-9] #This too

band_8a = glob.glob('*B8A.jp2')
band_4 = glob.glob('*B04.jp2')
band_3 = glob.glob('*B03.jp2')
band_2 = glob.glob('*B02.jp2')
band_list = band_8a+band_4+band_3+band_2
warp_options = gdal.WarpOptions(format="GTiff",resampleAlg="cubic", width=5490, height=5490)
for i in band_list:
    img = gdal.Open(i)
    gdal.Warp(str(i[-7:-4]) + '.tif', img, options = warp_options)

sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\gpu\Scripts') 
gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','gpu','Scripts','gdal_merge.py')
merge_command = ["python", gm,'-separate', "-o", file_name+"_NRGB.tif", "B8A.tif", "B04.tif","B03.tif","B02.tif"]
subprocess.call(merge_command,shell=True)

width = 5490
height = 5490
tilesize = 549

print('Tiling images...')
for i in range(0, width, tilesize):
    for j in range(0, height, tilesize):
        gdal.Translate(destName = 'Tiles/'+str(i)+'_'+str(j)+'_'+granule_tag+'.png',
                      srcDS = file_name+'_NRGB.tif',
                      srcWin = [i,j,tilesize,tilesize],
                      noData = None,
                      format = "PNG"
                      )
       
    
im_height = 512
im_width = 512
os.chdir('Tiles/')
ids = glob.glob('*.png')
for i in ids:
    if '.xml' in i:
        ids.remove(i)

try:
    ids.remove('Thumbs.db')
except:
    pass

print("No. of images = ", len(ids))

fixed_list = [0,9,1,2,3,4,5,6,7,8,10,19,11,12,13,14,15,16,17,18,20,29,21,22,23,24,25,26,27,28,
             30,39,31,32,33,34,35,36,37,38,40,49,41,42,43,44,45,46,47,48,50,59,51,52,53,54,55,56,57,58,
             60,69,61,62,63,64,65,66,67,68,70,79,71,72,73,74,75,76,77,78,80,89,81,82,83,84,85,86,87,88,
             90,99,91,92,93,94,95,96,97,98]

new_ids = [ids[i] for i in fixed_list]

X_test = np.zeros((len(ids), im_height, im_width, 4), dtype=np.float32)

print('Loading tiles for prediction...')

for n, id_ in tqdm_notebook(enumerate(new_ids), total=len(new_ids)):
    # Load images
    img = load_img(id_)
    x_img = img_to_array(img)
    #x_img = gdal_array.LoadFile(id_)
    x_img = resize(x_img, (im_height, im_width, 4), mode = 'edge', preserve_range = True)
    X_test[n] = x_img/256.0

print('Loading model...')
model = load_model(r'U:\Training_Data\CNN_Models\Model_123_whole.h5')

print('Predicting...')
t3 = time.time()
result = model.predict(X_test,batch_size=4)
t4 = time.time()
Total = round((t4-t3)/60, 1)
print('Predictions complete, took '+str(Total)+' minutes.')

result_new = result*10000

os.chdir('..')
os.chdir('Result/')

print('Saving results...')
for i,k in enumerate(new_ids):
    x_img = cv2.resize(result_new[i].astype('float32'), (549, 549))
    cv2.imwrite(str(k),x_img)

os.chdir('..')
os.chdir('Tiles/')
metadata_list = glob.glob('*.xml')
os.chdir('..')
dest1 = 'Result/'
source = 'Tiles/'


for f in metadata_list:
    shutil.copy2(source+f, dest1)

os.chdir('Result/')
mosaic_list = glob.glob('*.png')

print('Mosaicking results...')
gdal.Warp(destNameOrDestDS='_pred.tif',
          srcDSOrSrcDSTab=mosaic_list,
          resampleAlg='near',
          width=5490,
          height=5490,
          format='GTiff',
          outputType=gdal.GDT_Byte
          )

print('Sieving and buffering...')
gdal_path = r'C:\Users\jbrown\AppData\Local\conda\conda\envs\gpu\Scripts'
gdal_calc_path = os.path.join(gdal_path, 'gdal_calc.py')

input_file_path = '_pred.tif'
output_file_path = '_pred_filtered.tif'
calc_expr = '"A>=10.0"'
nodata = '0'
typeof = '"Byte"'

gdal_calc_str = 'python {0} -A {1} --outfile={2} --calc={3} --NoDataValue={4} --type={5}'
gdal_calc_process = gdal_calc_str.format(gdal_calc_path, input_file_path, 
    output_file_path, calc_expr, nodata, typeof)

os.system(gdal_calc_process)

gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','neural2','Scripts','gdal_sieve.py')
sieve_command = ["python", gm,'-st','10','-8','-nomask','-of','GTiff','_pred_filtered.tif','_sieved.tif']
subprocess.call(sieve_command,shell=True)

gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','gpu','Scripts','gdal_proximity.py')
prox_command = ["python", gm, '-of','GTiff','-ot','Byte','-distunits','PIXEL','-maxdist','9','-ot','Byte','-fixed-buf-val','1.0','_sieved.tif','_prox.tif']
subprocess.call(prox_command,shell=True)


sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\gpu\Scripts')
gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','gpu','Scripts','gdal_calc.py')
calc_command = ["python", gm,'--calc','A+B','-A','_prox.tif','-B','_sieved.tif','--NoDataValue','255','--outfile','final_test_123.tif',]
subprocess.call(calc_command,shell=True)

t1 = time.time()
Total = round((t1-t0)/60, 1)

os.chdir('..')
print('Mask complete!')
print('Algorithm took '+str(Total)+' total minutes to complete')
