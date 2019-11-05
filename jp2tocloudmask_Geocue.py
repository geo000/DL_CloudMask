from keras.models import load_model 
from osgeo import gdal, gdal_array
import sys, subprocess, os, glob, time
import numpy as np
import pandas as pd

t0 = time.time()

directory_with_jp2s = r'U:\Training_Data\jp2\apabs\T31PGM' #<---- directory with jp2s from multiple images
os.chdir(directory_with_jp2s)

image_list = glob.glob('*.jp2')
image_dict = {}

with open('U:\Sentinel_RPM_Load_List_T31PGM.txt','r') as loadlist: #<----Load list containing the unique image dates to be processed
    for i in loadlist:
        image_dict.update({i[:8]:[]})
        for image in image_list:
            if i[:8] in image:
                print(image)
                image_dict.setdefault(i[:8], []).append(image)
                
for x, i in image_dict.items():
    print('Starting. Resampling and converting jp2s...')
    for image_file in i:
        warp_options = gdal.WarpOptions(format="GTiff",resampleAlg="cubic", width=5490, height=5490,outputType=gdal.GDT_Float32)
        img = gdal.Open(file)
        gdal.Warp(str(file[-7:-4]) + '.tif', img, options = warp_options)

    print('reading xy coords from band 1...')
    gdal.Translate(destName='B01.xyz',srcDS='B01.tif',format="XYZ",outputType=gdal.GDT_Byte)
    xy = pd.read_csv('B01.xyz',delimiter='\s+',header=None,usecols=[0,1],names=['x','y'])

    print('calculating new raster values...')
    fileList = glob.glob('*B??.tif')
    for file in fileList:
        sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\neural2\Scripts')
        gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','neural2','Scripts','gdal_calc.py')
        merge_command = ["python", gm,"--outfile", file[0:3]+'_updated.tif', '-A' ,file,'--calc','A*0.0001']
        subprocess.call(merge_command,shell=True)

    df = pd.DataFrame(columns=['B01','B02','B03','B04','B05','B06','B07','B08','B8A','B09','B10','B11','B12'])
    fileList = glob.glob('*updated.tif')

    print('writing new values to dataframe...')
    for file in fileList:
        rasterArray = gdal_array.LoadFile(file)
        write_band = rasterArray.flatten()
        band_number = str(file[0:3])
        df[band_number] = write_band

    #split cell here ^
    #t0 = time.time()
    print('loading model...')
    model = load_model(r'U:\Training_Data\Models\Model_313.h5')

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

    print('sieving and buffering...')
    sys.path.append(r'C:\Users\jbrown\AppData\Local\conda\conda\envs\neural2\Scripts')
    gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','neural2','Scripts','gdal_sieve.py')
    sieve_command = ["python", gm,'-st','3','-8','-nomask','-of','GTiff','_pred.tif','_sieved.tif']
    subprocess.call(sieve_command,shell=True)

    gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','neural2','Scripts','gdal_proximity.py')
    prox_command = ["python", gm, '-of','GTiff','-distunits','PIXEL','-maxdist','8','-ot','Byte','-fixed-buf-val','1.0','_sieved.tif','_prox.tif']
    subprocess.call(prox_command,shell=True)

    gm = os.path.join('C:\\','Users','jbrown','AppData','Local','conda','conda','envs','neural2','Scripts','gdal_calc.py')
    merge_command = ["python", gm,"--outfile", image_file[:21]+"_313.tif",'-A',"_sieved.tif",'-B','_prox.tif','--calc','A+B']
    subprocess.call(merge_command,shell=True)

    t1 = time.time()
    Total = round((t1-t0)/60, 1)

    print('Mask complete!')
    print('Algorithm took '+str(Total)+' total minutes to complete')
