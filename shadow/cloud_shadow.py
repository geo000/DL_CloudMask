#single image shadow classification
import os, random, glob, sys, subprocess, time, cv2, datetime, shutil, contextlib, datetime, math, csv
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from skimage.io import imread
from skimage.transform import resize
import tifffile
from osgeo import gdal, gdal_array, osr, ogr
import xml.etree.ElementTree as ET

t0 = time.time()
st = datetime.datetime.fromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S')
print(st)

path_to_jp2s = r'D:\NewCloudTrials\Test\T31SDA_20201230'
gdal_path = r'C:\Users\jbrow\anaconda3\envs\test\Scripts'
os.chdir(path_to_jp2s)
file_name = path_to_jp2s[-15:]

band_list = glob.glob('*B02_corr.tif')[0]
blue = gdal.Open(band_list)
srs_prj = blue.GetProjection()
geo_transform = blue.GetGeoTransform()
blue = blue.GetRasterBand(1).ReadAsArray()
band_list = glob.glob('*B08_corr.tif')[0]
nir = gdal.Open(band_list)
nir = nir.GetRasterBand(1).ReadAsArray()
band_list = glob.glob('*B11_corr.tif')[0]
swir = gdal.Open(band_list)
swir = swir.GetRasterBand(1).ReadAsArray()

csi = (nir.astype('float32') + swir.astype('float32')) * 0.5
t3 = 0.6 #csi threshold coefficient, larger = more cloud shadows detected
t4 = 0.6 #smaller number removes influence of water
x = 0

if csi.min() < 0:
    x = 0.150
else:
    x = csi.min()

T3 = x + t3 * (csi.mean() - x)
T4 = blue.min() + t4 * (blue.mean() - blue.min())

csi[csi < T3] = 1
csi[csi > 1] = 0
blue[blue < T4] = 1
blue[blue > 1] = 0

shadows = csi + blue
shadows[shadows < 2] = 0
shadows[shadows > 0] = 1

driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create('shadows.tif',
                       10980,
                       10980,
                       1,
                       gdal.GDT_Int16,
                       options=["COMPRESS=LZW"])

srs = osr.SpatialReference()
srs.ImportFromEPSG(int(srs_prj[-8:-3]))
dst_ds.SetGeoTransform(list(geo_transform))
dst_ds.SetProjection(srs.ExportToWkt())
dst_band = dst_ds.GetRasterBand(1)
#dst_band.SetNoDataValue(0)
dst_band.WriteArray(shadows)
dst_ds = None

band_list = glob.glob('shadows.tif')[0]
shadows = gdal.Open(band_list)
shadow_stats = shadows.GetRasterBand(1).ReadAsArray()
total_shadow_pixels = shadow_stats.sum()

metadata = glob.glob('*TL.xml')[0]
tree = ET.parse(metadata)
root = tree.getroot()
zenith = float(root[1][1][1][0].text)
azimuth = float(root[1][1][1][1].text)

def shadow_matrix_values(zenith, azimuth, cloud_height = 2200):
    if azimuth < 90:
        new_azimuth = 90 - azimuth
    if 180 > azimuth > 90:
        new_azimuth = azimuth - 90

    if 270 > azimuth > 180:
        new_azimuth = azimuth - 180

    if 360 > azimuth > 270:
        new_azimuth = azimuth - 270

    solar_altitude = np.deg2rad(90 - zenith)
    c = cloud_height / np.tan(solar_altitude)
    rise = np.sin(np.deg2rad(new_azimuth)) * c
    a_squared = (c*c) - (rise*rise)
    run = math.sqrt(a_squared)

    rise = round(rise / 20)
    run = round(run / 20)

    if azimuth <= 90:
        run *= -1

    if 180 > azimuth > 90:
        rise *= -1
        run *= -1

    if 270 > azimuth > 180:
        rise *= -1

    if 360 > azimuth > 270:
        pass

    return rise, run

cloud_lowermost = 20
cloud_uppermost = 2400

cloud_shadow_curve = []

for i,j in enumerate(range(cloud_lowermost,cloud_uppermost,100)):

    rise, run = shadow_matrix_values(zenith, azimuth,cloud_height=j)

    cloudmask = glob.glob('*cloudmask.tif')[0]
    img = cv2.imread(cloudmask,0)
    rows,cols = img.shape

    M = np.float32([[1,0,run],[0,1,rise]])
    shadows2 = cv2.warpAffine(img,M,(cols,rows),borderValue = 255)

    band_list = glob.glob('*B11.jp2')
    swir = gdal.Open(band_list[0])
    srs_prj = swir.GetProjection()
    geo_transform = swir.GetGeoTransform()
    #cloudmask = gdal.Open(cloudmask)

    in_raster = 'shadow_match_file_'+str(i)+'_delete.tif'
    driver = gdal.GetDriverByName("GTiff")
    dst_ds = driver.Create(in_raster,5490,5490,1,gdal.GDT_Byte)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(int(srs_prj[-8:-3]))
    dst_ds.SetGeoTransform(list(geo_transform))
    dst_ds.SetProjection(srs.ExportToWkt())
    dst_band = dst_ds.GetRasterBand(1)
    dst_band.SetNoDataValue(255)
    dst_band.WriteArray(shadows2)
    dst_ds = None
    gdal_poly_path = os.path.join(gdal_path,'gdal_polygonize.py')
    out_shape = 'masked_shadow_match'+str(i)+'_delete.shp'
    formatType = '"ESRI Shapefile"'
    gdal_poly_str = 'python {0} -8 {1} -mask {2} -f {3} {4}'
    gdal_poly_process = gdal_poly_str.format(gdal_poly_path, in_raster, in_raster, formatType, out_shape)
    os.system(gdal_poly_process)

    def boundingBoxToOffsets(bbox, geot):
        col1 = int((bbox[0] - geot[0]) / geot[1])
        col2 = int((bbox[1] - geot[0]) / geot[1]) + 1
        row1 = int((bbox[3] - geot[3]) / geot[5])
        row2 = int((bbox[2] - geot[3]) / geot[5]) + 1
        return [row1, row2, col1, col2]


    def geotFromOffsets(row_offset, col_offset, geot):
        new_geot = [
        geot[0] + (col_offset * geot[1]),
        geot[1],
        0.0,
        geot[3] + (row_offset * geot[5]),
        0.0,
        geot[5]
        ]
        return new_geot


    def setFeatureStats(fid, min, max, mean, median, sd, sum, count, names=["min", "max", "mean", "median", "sd", "sum", "count", "id"]):
        featstats = {
        names[0]: min,
        names[1]: max,
        names[2]: mean,
        names[3]: median,
        names[4]: sd,
        names[5]: sum,
        names[6]: count,
        names[7]: fid,
        }
        return featstats

    mem_driver = ogr.GetDriverByName("Memory")
    mem_driver_gdal = gdal.GetDriverByName("MEM")
    shp_name = "temp"

    fn_raster = 'shadows.tif'
    fn_zones = out_shape

    r_ds = gdal.Open(fn_raster)
    p_ds = ogr.Open(fn_zones)

    lyr = p_ds.GetLayer()
    geot = r_ds.GetGeoTransform()
    nodata = r_ds.GetRasterBand(1).GetNoDataValue()

    zstats = []

    p_feat = lyr.GetNextFeature()
    niter = 0

    while p_feat:
        if p_feat.GetGeometryRef() is not None:
            if os.path.exists(shp_name):
                mem_driver.DeleteDataSource(shp_name)
            tp_ds = mem_driver.CreateDataSource(shp_name)
            tp_lyr = tp_ds.CreateLayer('polygons', None, ogr.wkbPolygon)
            tp_lyr.CreateFeature(p_feat.Clone())
            offsets = boundingBoxToOffsets(p_feat.GetGeometryRef().GetEnvelope(),\
            geot)
            new_geot = geotFromOffsets(offsets[0], offsets[2], geot)

            tr_ds = mem_driver_gdal.Create(\
            "", \
            offsets[3] - offsets[2], \
            offsets[1] - offsets[0], \
            1, \
            gdal.GDT_Byte)

            tr_ds.SetGeoTransform(new_geot)
            gdal.RasterizeLayer(tr_ds, [1], tp_lyr, burn_values=[1])
            tr_array = tr_ds.ReadAsArray()

            r_array = r_ds.GetRasterBand(1).ReadAsArray(\
            offsets[2],\
            offsets[0],\
            offsets[3] - offsets[2],\
            offsets[1] - offsets[0])

            id = p_feat.GetFID()

            if r_array is not None:
                maskarray = np.ma.MaskedArray(\
                r_array,\
                maskarray=np.logical_or(r_array==nodata, np.logical_not(tr_array)))

                if maskarray is not None:
                    zstats.append(setFeatureStats(\
                    id,\
                    maskarray.min(),\
                    maskarray.max(),\
                    maskarray.mean(),\
                    np.ma.median(maskarray),\
                    maskarray.std(),\
                    maskarray.sum(),\
                    maskarray.count()))
                else:
                    zstats.append(setFeatureStats(\
                    id,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata))
            else:
                zstats.append(setFeatureStats(\
                    id,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata,\
                    nodata))

            tp_ds = None
            tp_lyr = None
            tr_ds = None

            p_feat = lyr.GetNextFeature()

    fn_csv = "zstats_"+str(i)+'_delete.csv'
    col_names = zstats[0].keys()
    with open(fn_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, col_names)
        writer.writeheader()
        writer.writerows(zstats)
        
    df = pd.read_csv(fn_csv)
    target_shadow_pixels = (df['sum']).sum()
    total_target_percentage = (target_shadow_pixels / total_shadow_pixels)
    cloud_shadow_curve.append(total_target_percentage)

m = max(cloud_shadow_curve)
shadow_matcher_max = [i for i, j in enumerate(cloud_shadow_curve) if j == m]
shadow_matcher_max = (shadow_matcher_max[0])

fileList = []
for i in range(0,shadow_matcher_max+1):
    elem = glob.glob('shadow_match_file_'+str(i)+'_delete.tif')[0]
    fileList.append(elem)

#fileList= glob.glob('*shadow_match_file_*.tif')
listtoStr = ' '.join([str(elem)] for elem in fileList)

gdal_merge_path = os.path.join(gdal_path,'gdal_merge.py')
pixel_size = '10'
outputFile = 'shadowmask_matcher_final.tif'

gdal_merge_str = 'python {0} -ps {1} {2} -o {3} '+listtoStr
gdal_merge_process = gdal_merge_str.format(gdal_merge_path, pixel_size, pixel_size, outputFile)
os.system(gdal_merge_process)

for i in fileList:
    os.remove(i)

warp_shadow_match = gdal.Open('shadowmask_matcher_final.tif')
srs_prj = warp_shadow_match.GetProjection()
geo_transform = warp_shadow_match.GetGeoTransform()
warp_shadow_match = warp_shadow_match.GetRasterBand(1).ReadAsArray()

warp_shadow_match[warp_shadow_match > 1] = 0
matched_batch = warp_shadow_match.astype('int16') + shadows.astype('int16')
matched_batch[matched_batch < 2] = 0
matched_batch[matched_batch > 0] = 1

driver = gdal.GetDriverByName("GTiff")
dst_ds = driver.Create('shadows_final.tif',
                       10980,
                       10980,
                       1,
                       gdal.GDT_Int16,
                       options=["COMPRESS=LZW"])

srs = osr.SpatialReference()
srs.ImportFromEPSG(int(srs_prj[-8:-3]))
dst_ds.SetGeoTransform(list(geo_transform))
dst_ds.SetProjection(srs.ExportToWkt())
dst_band = dst_ds.GetRasterBand(1)
dst_band.SetNoDataValue(0)
dst_band.WriteArray(matched_batch)
dst_ds = None

#Sieve
gdal_sieve_path = os.path.join(gdal_path,'gdal_sieve.py')
input_file_path = 'shadows_final.tif'
output_file_path = '_sieved.tif'
threshold = '10' #Pixel clusters 10 and under will be removed
connectedness = '-8' #Pixel cluster alg will consider pixels on the 4 corners and 4 edges
outputFormat = 'GTiff'
nomask = '-nomask'
typeof = '"Byte"'

gdal_sieve_str = 'python {0} -st {1} {2} {3} -of {4} {5} {6}'
gdal_sieve_process = gdal_sieve_str.format(gdal_sieve_path, threshold, connectedness, nomask, outputFormat, input_file_path, output_file_path)
os.system(gdal_sieve_process)

#Proximity
gdal_prox_path = os.path.join(gdal_path,'gdal_proximity.py')
input_file_path = '_sieved.tif'
output_file_path = '_prox.tif'
dist_units = 'PIXEL' #The buffer applied will be measured by pixels, as opposed to meters
max_dist = '6' #Cloud buffer amount.  Note that these are 10m pixels, so a 3 pixel buffer = 30m buffer.
fixed_buffer_val = '1.0'

gdal_prox_str = 'python {0} -of {1} -distunits {2} -maxdist {3} -ot {4} -fixed-buf-val {5} {6} {7}'
gdal_prox_process = gdal_prox_str.format(gdal_prox_path,outputFormat,dist_units,max_dist,typeof,fixed_buffer_val,input_file_path,output_file_path)
os.system(gdal_prox_process)

#Add proximity and sieve, creating final mask
gdal_calc_path = os.path.join(gdal_path, 'gdal_calc.py')
out_mask = '_shadowmask.img'
A = '_prox.tif'
B = '_sieved.tif'
calc_expr = '"A+B"' #Buffer layer created by gdal_proximity is produced without original data, so it must be combined with gdal_sieve output
creation_option = 'COMPRESSED=YES'
nodata = '255'
outputFormat = 'HFA'

gdal_calc_str = 'python {0} --calc {1} -A {2} -B {3} --creation-option {4} --NoDataValue {5} --format {6} --type {7} --outfile {8}'
gdal_calc_process = gdal_calc_str.format(gdal_calc_path,calc_expr,A,B,creation_option,nodata,outputFormat,typeof,out_mask)
os.system(gdal_calc_process)


print('done!')
