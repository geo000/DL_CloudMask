import glob
import os
from osgeo import gdal
import glob

os.chdir(r'U:\Training_Data\jp2\T37TEE_20190806')

fileList= glob.glob('*B??.jp2')

warp_options = gdal.WarpOptions(format="GTiff",resampleAlg="cubic", width=5490, height=5490)

for i in fileList:
    print(i)
    img = gdal.Open(i)
    gdal.Warp(str(i[-7:-4]) + '.tif', img, options = warp_options)

import sys, os, subprocess
sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\gdal\Scripts') 
gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','gdal','Scripts','gdal_merge.py')
merge_command = ["python", gm,'-separate', "-o", i[:15]+".tif", "B01.tif", "B02.tif","B03.tif","B04.tif",
                 "B05.tif","B06.tif","B07.tif","B08.tif","B8A.tif","B09.tif","B10.tif","B11.tif","B12.tif",]
subprocess.call(merge_command,shell=True)
fileList=''
gm=''

removeList = glob.glob('B*.tif')
for i in removeList:
    os.remove(i)
    
print('Done!')
