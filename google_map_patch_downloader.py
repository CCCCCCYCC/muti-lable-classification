# This code is used to download images from Google Maps
# @date  : 2020-3-13
# @author: Zheng Jie
# @E-mail: zhengjie9510@qq.com
# @ https://github.com/zhengjie9510/google-map-downloader

import io
import math
import random
import multiprocessing
import time
import urllib.request as ur
from math import floor, pi, log, tan, atan, exp
from threading import Thread
import imghdr
import PIL.Image as pil
import cv2
import numpy as np
from osgeo import gdal, osr
from pyproj import Transformer


# ------------------Interchange between WGS-84 and Web Mercator-------------------------
# WGS-84 to Web Mercator
def wgs_to_mercator(Lat_x, Lon_y):
    # y = 85.0511287798 if y > 85.0511287798 else y
    # y = -85.0511287798 if y < -85.0511287798 else y
    #
    # x2 = x * 20037508.34 / 180
    # y2 = log(tan((90 + y) * pi / 360)) / (pi / 180)
    # y2 = y2 * 20037508.34 / 180

    wgs84_TO_Mercator = Transformer.from_crs("epsg:4326", "epsg:3857")  # wgs84 TO Mercator
    w2m = wgs84_TO_Mercator.transform(Lat_x, Lon_y)  # Lat Lon
    x = w2m[0]
    y = w2m[1]

    return x, y


# Web Mercator to WGS-84
def mercator_to_wgs(Lon_y, Lat_x):
    # x2 = x / 20037508.34 * 180
    # y2 = y / 20037508.34 * 180
    # y2 = 180 / pi * (2 * atan(exp(y2 * pi / 180)) - pi / 2)

    Mercator_TO_wgs84 = Transformer.from_crs("epsg:3857", "epsg:4326")  # Mercator TO wgs84
    m2w = Mercator_TO_wgs84.transform(Lon_y, Lat_x)  # Lon Lat

    y = m2w[0]
    x = m2w[1]

    return x, y


# ----------------------------------------------------------------------------------------------
# https://stackoverflow.com/questions/37464824/converting-longitude-latitude-to-tile-coordinates
# https://developers.google.com/maps/documentation/javascript/examples/map-coordinates
# ----------------convert the wgs84 coordinates to the x,y pixel coordinates---------------------
EARTH_RADIUS = 6378137
# MAX_LATITUDE = 85.0511287798
MAX_LATITUDE = 85.0511287798


def project(lat, lon):
    d = math.pi / 180
    _max = MAX_LATITUDE
    # lat value
    if min([_max, lat]) > _max:
        lat = min([_max, lat])
    elif min([_max, lat]) <= _max:
        lat = min([_max, lat])
    sin = math.sin(lat * d)
    return {'x': EARTH_RADIUS * lon * d,
            'y': EARTH_RADIUS * math.log((1 + sin) / (1 - sin)) / 2}


def zoomScale(zoom):
    pass
    return 256 * math.pow(2, zoom)


def transform(point, scale):
    scale = scale or 1
    point['x'] = math.floor(scale * (2.495320233665337e-8 * point['x'] + 0.5))
    point['y'] = math.floor(scale * (-2.495320233665337e-8 * point['y'] + 0.5))
    return point


# --------------------------------------------------------------------------------------

# -----------------Interchange between GCJ-02 to WGS-84---------------------------
# All public geographic data in mainland China need to be encrypted with GCJ-02, introducing random bias
# This part of the code is used to remove the bias
def transformLat(x, y):
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
    return ret


def transformLon(x, y):
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
    ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
    ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
    ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
    return ret


def delta(lat, lon):
    '''
    Krasovsky 1940
    //
    // a = 6378245.0, 1/f = 298.3
    // b = a * (1 - f)
    // ee = (a^2 - b^2) / a^2;
    '''
    a = 6378245.0  # a: Projection factor of satellite ellipsoidal coordinates projected onto a flat map coordinate system
    ee = 0.00669342162296594323  # ee: Eccentricity of ellipsoid
    dLat = transformLat(lon - 105.0, lat - 35.0)
    dLon = transformLon(lon - 105.0, lat - 35.0)
    radLat = lat / 180.0 * math.pi
    magic = math.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = math.sqrt(magic)
    dLat = (dLat * 180.0) / ((a * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLon = (dLon * 180.0) / (a / sqrtMagic * math.cos(radLat) * math.pi)
    return {'lat': dLat, 'lon': dLon}


def outOfChina(lat, lon):
    if (lon < 72.004 or lon > 137.8347):
        return True
    if (lat < 0.8293 or lat > 55.8271):
        return True
    return False


def gcj_to_wgs(gcjLon, gcjLat):
    if outOfChina(gcjLat, gcjLon):
        return (gcjLon, gcjLat)
    d = delta(gcjLat, gcjLon)
    return (gcjLon - d["lon"], gcjLat - d["lat"])


def wgs_to_gcj(wgsLon, wgsLat):
    if outOfChina(wgsLat, wgsLon):
        return wgsLon, wgsLat
    d = delta(wgsLat, wgsLon)
    return wgsLon + d["lon"], wgsLat + d["lat"]


# --------------------------------------------------------------

# ---------------------------------------------------------
# Get tile coordinates in Google Maps based on latitude and longitude of WGS-84
def wgs_to_tile(j, w, z):  # (Lon, Lat, Zoom) # Check
    '''
    Get google-style tile cooridinate from geographical coordinate
    j : Longittude
    w : Latitude
    z : zoom
    '''
    isnum = lambda x: isinstance(x, int) or isinstance(x, float)
    if not (isnum(j) and isnum(w)):
        raise TypeError("j and w must be int or float!")

    if not isinstance(z, int) or z < 0 or z > 22:
        raise TypeError("z must be int and between 0 to 22.")

    if j < 0:
        j = 180 + j
    else:
        j += 180
    j /= 360  # make j to (0,1)

    w = 85.0511287798 if w > 85.0511287798 else w
    w = -85.0511287798 if w < -85.0511287798 else w
    w = log(tan((90 + w) * pi / 360)) / (pi / 180)
    w /= 180  # make w to (-1,1)
    w = 1 - (w + 1) / 2  # make w to (0,1) and left top is 0-point

    num = 2 ** z
    x = floor(j * num)
    y = floor(w * num)
    return x, y


def pixls_to_mercator(zb):
    # Get the web Mercator projection coordinates of the four corners of the area according to the four corner coordinates of the tile
    inx, iny = zb["LT"]  # left top
    inx2, iny2 = zb["RB"]  # right bottom
    length = 20037508.3427892
    sum = 2 ** zb["z"]
    LTx = inx / sum * length * 2 - length
    LTy = -(iny / sum * length * 2) + length

    RBx = (inx2 + 1) / sum * length * 2 - length
    RBy = -((iny2 + 1) / sum * length * 2) + length

    # LT=left top,RB=right bottom
    # Returns the projected coordinates of the four corners
    res = {'LT': (LTx, LTy), 'RB': (RBx, RBy),
           'LB': (LTx, RBy), 'RT': (RBx, LTy)}
    return res


def corner_tile_coordinates(center_x, center_y):
    # Get tile coordinates for four corner points
    LT = (center_x - 1, center_y - 1)
    RT = (center_x + 1, center_y - 1)
    LB = (center_x - 1, center_y + 1)
    RB = (center_x + 1, center_y + 1)

    # LT=left top,RB=right bottom
    # Returns the tile coordinates of the four corners
    res = {'LT': LT, 'RT': RT, 'LB': LB, 'RB': RB}
    print("res", res)
    return res


def tile_to_pixls(zb):
    # Tile coordinates are converted to pixel coordinates of the four corners
    out = {}
    width = (zb["RT"][0] - zb["LT"][0] + 1) * 256
    height = (zb["LB"][1] - zb["LT"][1] + 1) * 256
    out["LT"] = (0, 0)
    out["RT"] = (width, 0)
    out["LB"] = (0, -height)
    out["RB"] = (width, -height)
    return out


# -----------------------------------------------------------

# ---------------------------------------------------------
class Downloader(Thread):
    # multiple threads downloader
    def __init__(self, index, count, urls, datas):
        # index represents the number of threads
        # count represents the total number of threads
        # urls represents the list of URLs nedd to be downloaded
        # datas represents the list of data need to be returned.
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count

    def download(self, url):
        HEADERS = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0'}
        request = ur.Request(url, headers=HEADERS)
        err = 0
        while err < 3:
            try:
                req = ur.Request(url, headers=HEADERS)
                with ur.urlopen(req, timeout=15) as response:
                    # Priority check HTTP status code
                    if response.status != 200:
                        print(f"Skip HTTP {response.status} error: {url}")
                        return None

                    data = response.read()

                    # Verify data validity
                    if len(data) < 1024 or not imghdr.what(None, data):
                        print(f"Invalid response data: {url}")
                        return None

                    return data

            except ur.HTTPError as e:
                if e.code == 404:
                    print(f"Map tile does not exist: {url}")
                else:
                    print(f"HTTP error {e.code}: {url}")
                return None
            except Exception as e:
                print(f"Network error: {str(e)} - {url}")
                return None

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue
            self.datas[i] = self.download(url)


# ---------------------------------------------------------

# ---------------------------------------------------------
def getExtent(center_Lat_x, center_Lon_y, z, source="Google"):
    pos_x, pos_y = wgs_to_tile(center_Lon_y, center_Lat_x, z)  # (Lon, Lat, Zoom)
    print("pos x y", pos_x, pos_y)
    Xframe = corner_tile_coordinates(pos_x, pos_y)
    print("LT tile position {}, RT tile position {}, LB tile position {}, RB tile position {}".format(Xframe["LT"],
                                                                                                      Xframe["LB"],
                                                                                                      Xframe["RT"],
                                                                                                      Xframe["RB"]))
    for i in ["LT", "LB", "RT", "RB"]:
        Xframe[i] = mercator_to_wgs(*Xframe[i])
    if source == "Google":
        pass
    elif source == "Google China":
        for i in ["LT", "LB", "RT", "RB"]:
            Xframe[i] = gcj_to_wgs(*Xframe[i])
    else:
        raise Exception("Invalid argument: source.")
    return Xframe


def saveTiff(r, g, b, gt, filePath):
    fname_out = filePath
    driver = gdal.GetDriverByName('GTiff')
    # Create a 3-band dataset
    dset_output = driver.Create(fname_out, r.shape[1], r.shape[0], 3, gdal.GDT_Byte)
    dset_output.SetGeoTransform(gt)
    try:
        proj = osr.SpatialReference()
        proj.ImportFromEPSG(4326)
        dset_output.SetSpatialRef(proj)
    except:
        print("Error: Coordinate system setting failed")
    dset_output.GetRasterBand(1).WriteArray(r)
    dset_output.GetRasterBand(2).WriteArray(g)
    dset_output.GetRasterBand(3).WriteArray(b)
    dset_output.FlushCache()
    dset_output = None
    print("Image Saved")


# ---------------------------------------------------------

# ---------------------------------------------------------
MAP_URLS = {
    "Google": "http://mts0.googleapis.com/vt?lyrs={style}&x={x}&y={y}&z={z}",
    "Google China": "http://mt2.google.cn/vt/lyrs={style}&hl=zh-CN&gl=CN&src=app&x={x}&y={y}&z={z}"}


def get_url(source, x, y, z, style):  #
    if source == 'Google China':
        url = MAP_URLS["Google China"].format(x=x, y=y, z=z, style=style)
    elif source == 'Google':
        url = MAP_URLS["Google"].format(x=x, y=y, z=z, style=style)
    else:
        raise Exception("Unknown Map Source ! ")
    return url


def get_urls(Lat_x, Lon_y, z, source, style):
    pos_x, pos_y = wgs_to_tile(Lon_y, Lat_x, z)  # convert WGS84 to tile coordinates
    print("center_tile_pos:", pos_x, pos_y)
    print("center_tile_pixel_index:", pos_x * 256, pos_y * 256)
    four_corner_tile = corner_tile_coordinates(pos_x, pos_y)
    print("cor_tile", four_corner_tile)
    # res = {'LT': LT, 'RT': RT, 'LB': LB, 'RB': RB}
    LT = four_corner_tile['LT']  # top left
    RB = four_corner_tile['RB']  # bottom right
    lenx = RB[0] - LT[0] + 1
    leny = RB[1] - LT[1] + 1
    print("Total tiles number：{x} X {y}".format(x=lenx, y=leny))
    print("range y:", LT[1], LT[1] + leny)
    print("range x:", LT[0], LT[0] + lenx)
    urls = [get_url(source, i, j, z, style) for j in range(LT[1], LT[1] + leny) for i in range(LT[0], LT[0] + lenx)]
    return urls


# ---------------------------------------------------------
def wgs_to_tile(lon, lat, zoom):
    lat = max(min(lat, 85.0511287798), -85.0511287798)

    if lon < 0:
        lon += 180
    else:
        lon += 180
    lon /= 360

    lat_rad = math.radians(lat)
    lat_merc = math.log(tan(lat_rad) + 1.0 / math.cos(lat_rad)) / pi
    lat = (1.0 - lat_merc) / 2

    num = 2 ** zoom
    x = floor(lon * num)
    y = floor(lat * num)

    # Verify coordinate validity
    max_tile = 2 ** zoom - 1
    if not (0 <= x <= max_tile and 0 <= y <= max_tile):
        raise ValueError(f"Invalid tile coordinates: x={x}, y={y} (z={zoom})")

    return x, y


# ---------------------------------------------------------
def merge_tiles(datas, Lat_x, Lon_y, z):
    pos_x, pos_y = wgs_to_tile(Lat_x, Lon_y, z)  # convert WGS84 to tile coordinates
    four_corner_tile = corner_tile_coordinates(pos_x, pos_y)
    LT = four_corner_tile['LT']  # top left
    RB = four_corner_tile['RB']  # bottom right
    lenx = RB[0] - LT[0] + 1
    leny = RB[1] - LT[1] + 1
    outpic = pil.new('RGBA', (lenx * 256, leny * 256))
    for i, data in enumerate(datas):
        picio = io.BytesIO(data)
        small_pic = pil.open(picio)
        y, x = i // lenx, i % lenx
        outpic.paste(small_pic, (x * 256, y * 256))
    print('Tiles merge completed')
    return outpic


def download_tiles(urls, multi=10):
    url_len = len(urls)
    datas = [None] * url_len
    if multi < 1 or multi > 20 or not isinstance(multi, int):
        raise Exception("multi of Downloader shuold be int and between 1 to 20.")
    tasks = [Downloader(i, multi, urls, datas) for i in range(multi)]
    for i in tasks:
        i.start()
    for i in tasks:
        i.join()
    return datas


def readTif(tif_file_path):
    dataset = gdal.Open(tif_file_path)
    if dataset == None:
        print('can not open the tif file')
    im_width = dataset.RasterXSize  # number of columns in the raster matrix
    im_height = dataset.RasterYSize  # number of rows in the raster matrix
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # get data

    return dataset, im_data


def safe_merge_tiles(datas, Lat_x, Lon_y, z):
    try:
        pos_x, pos_y = wgs_to_tile(Lon_y, Lat_x, z)
    except ValueError as e:
        print(f"Coordinate verification failed: {str(e)}")
        return pil.new('RGBA', (768, 768))

    try:
        corners = corner_tile_coordinates(pos_x, pos_y)
        LT = corners['LT']
        RB = corners['RB']
        lenx = RB[0] - LT[0] + 1
        leny = RB[1] - LT[1] + 1
    except:
        print("Tile coordinate calculation error, using default size")
        lenx, leny = 3, 3  # default 3x3 tiles

    outpic = pil.new('RGBA', (lenx * 256, leny * 256))

    valid_count = 0
    for i, data in enumerate(datas):
        if data is None:
            continue

        try:
            with io.BytesIO(data) as picio:
                small_pic = pil.open(picio).convert('RGBA')
                y_pos = i // lenx
                x_pos = i % lenx
                outpic.paste(small_pic, (x_pos * 256, y_pos * 256))
                valid_count += 1
        except Exception as e:
            print(f"Skip corrupted tile {i}: {str(e)[:50]}")

    print(f'Successfully merged {valid_count}/{len(datas)} tiles')
    return outpic


# ---------------------------------------------------------
class RobustDownloader(Thread):
    def __init__(self, index, count, urls, datas):
        super().__init__()
        self.urls = urls
        self.datas = datas
        self.index = index
        self.count = count

    def download(self, url):
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.54 Safari/537.36',
            'Referer': 'https://www.google.com/maps/'
        }

        for retry in range(3):
            try:
                req = ur.Request(url, headers=headers)
                with ur.urlopen(req, timeout=10) as response:
                    if response.status != 200:
                        print(f"Skip HTTP {response.status} error: {url}")
                        return None

                    data = response.read()
                    if len(data) < 1024 or not imghdr.what(None, data):
                        print(f"Invalid image data: {url}")
                        return None

                    return data

            except Exception as e:
                print(f"Download failed ({retry + 1}/3): {str(e)[:50]} - {url}")
                time.sleep(random.uniform(0.5, 1.5))

        return None

    def run(self):
        for i, url in enumerate(self.urls):
            if i % self.count != self.index:
                continue

            # Add random delay to prevent blocking
            time.sleep(random.uniform(0.1, 0.5))

            self.datas[i] = self.download(url)


def main(center_Lat, center_Lon, zoom, filePath, style='s', server="Google"):
    # Verify coordinates when generating URLs
    try:
        urls = get_urls(center_Lat, center_Lon, zoom, server, style)
    except ValueError as e:
        print(f"Initial coordinates invalid: {str(e)}")
        return

    # Filter invalid URLs
    valid_urls = []
    for url in urls:
        try:
            parts = url.split('&')
            x = int(parts[2].split('=')[1])
            y = int(parts[3].split('=')[1])
            if not (0 <= y < 2 ** zoom):
                raise ValueError
            valid_urls.append(url)
        except:
            print(f"Skip invalid tile URL: {url}")

    print(f"Valid tile count: {len(valid_urls)}/{len(urls)}")

    # Download data
    datas = [None] * len(valid_urls)
    workers = min(4, len(valid_urls))  # limit maximum threads
    tasks = [RobustDownloader(i, workers, valid_urls, datas) for i in range(workers)]

    for t in tasks:
        t.start()
    for t in tasks:
        t.join()

    # Merge and save
    outpic = safe_merge_tiles(datas, center_Lat, center_Lon, zoom)
    if outpic:
        outpic = outpic.convert('RGB')
        r, g, b = cv2.split(np.array(outpic))
        saveTiff(r, g, b, get_geotransform(center_Lat, center_Lon, zoom), filePath)


def get_geotransform(lat, lon, zoom):
    # Simplified geotransform generation
    scale = 20037508.34 * 2 / (2 ** zoom)
    return (
        lon - scale / 2,  # top left longitude
        scale / 256,  # longitude resolution
        0,
        lat + scale / 2,  # top left latitude
        0,
        -scale / 256  # latitude resolution
    )


def get_patch_from_GM(center_Lat, center_Lon, zoom, tif_save_path, patch_save_root):
    # 1. Download_tif_image
    main(center_Lat, center_Lon, zoom, tif_save_path, server="Google")

    # 2. Crop_the_tif_to_512 * 512
    dataset, im_data = readTif(tif_save_path)  # im_data is numpy array
    tile_index_x, tile_index_y = wgs_to_tile(center_Lon, center_Lat,
                                             zoom)  # get the index of the center point (Lon, Lat, Zoom)
    print("tile_index_x {} and tile_index_y {}".format(tile_index_x, tile_index_y))
    corners = corner_tile_coordinates(tile_index_x,
                                      tile_index_y)  # calculate the tile index of corners (return corners index:{'LT','RT','LB','RB'})
    print("dict corners:", corners)

    # Convert the center point to pixel index
    point = project(center_Lat, center_Lon)
    scaledZoom = zoomScale(zoom)
    point = transform(point, scaledZoom)
    print("center_point_pixel_index {} and {}".format(point['x'], point['y']))

    # Convert the corner point to pixel index
    corners['LT'] = (corners['LT'][0] * 256, corners['LT'][1] * 256)  # Origin (0, 0)
    corners['RT'] = ((corners['RT'][0] + 1) * 256, corners['RT'][1] * 256)
    corners['LB'] = (corners['LB'][0] * 256, (corners['LB'][1] + 1) * 256)
    corners['RB'] = ((corners['RB'][0] + 1) * 256, (corners['RB'][1] + 1) * 256)

    # Calculate the relative location between each real-corner and the center point
    Origin_LT = (corners['LT'][0] - corners['LT'][0], corners['LT'][1] - corners['LT'][1])
    RT = (corners['RT'][0] - corners['LT'][0], corners['RT'][1] - corners['LT'][1])
    LB = (corners['LB'][0] - corners['LT'][0], corners['LB'][1] - corners['LT'][1])
    RB = (corners['RB'][0] - corners['LT'][0], corners['RB'][1] - corners['LT'][1])
    Center_pt = (point["x"] - corners['LT'][0], point["y"] - corners['LT'][1])
    print("Origin_LT {}; RT {}; LB {}; RB {}; Center_pt {}".format(Origin_LT, RT, LB, RB, Center_pt))

    # 3. Crop the image to right 512 * 512
    # Calculate the 4 corner pixel position of 512 patch  ||  ** decide the size of the patch **
    Patch_LT = (Center_pt[0] - 256, Center_pt[1] - 256)
    Patch_RT = (Center_pt[0] + 256, Center_pt[1] - 256)
    Patch_LB = (Center_pt[0] - 256, Center_pt[1] + 256)
    Patch_RB = (Center_pt[0] + 256, Center_pt[1] + 256)
    print("Coordinates of the four point P_LT {}, P_RT {}, P_LB {}, P_RB {}".format(Patch_LT, Patch_RT, Patch_LB,
                                                                                    Patch_RB))

    # Calculate the difference between 512 patch and 768 original image
    Top_slice = abs(Origin_LT[1] - Patch_LT[1])  # y axis
    Left_slice = abs(Origin_LT[0] - Patch_LT[0])  # y axis
    Right_slice = abs(RB[0] - Patch_RB[0])  # x axis
    Bottom_slice = abs(RB[1] - Patch_RB[1])  # x axis
    print("Difference: Top {}, Bottom {}, Left {}, Right {}".format(Top_slice, Bottom_slice, Left_slice, Right_slice))

    # Crop the remote-sens numpy into 512 numpy array by using the slice method in the numpy
    print("The original rs_img size is {}", im_data.shape)  # (3, 768, 768)
    im_data_Transpose = im_data.transpose(1, 2, 0)  # (3, 768, 768) -> (768, 768, 3)
    im_data = im_data_Transpose[Top_slice:768 - Bottom_slice, Left_slice:768 - Right_slice, :]
    print("Final image shape:", im_data.shape)
    im = pil.fromarray(im_data)
    im.save(patch_save_root + '/' + 'Lat_' + str(center_Lat) + '_' + 'Lon_' + str(center_Lon) + '.png')
    end_time = time.time()
    print('lasted a total of {:.2f} seconds'.format(end_time - start_time))


# ---------------------------------------------------------
if __name__ == '__main__':
    start_time = time.time()

    tif_save_path = r'D:\muti-label\test.tif'
    patch_save_root = r'D:\muti-label\Images'

    # center point parameter
    dict_try = [
        # Add your coordinate points here
    ]

    for i in dict_try:
        center_Lon, center_Lat, zoom = i
        get_patch_from_GM(center_Lat=center_Lat,
                          center_Lon=center_Lon,
                          zoom=18,
                          tif_save_path=tif_save_path,
                          patch_save_root=patch_save_root)