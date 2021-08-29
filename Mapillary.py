import asyncio
import mercantile, mapbox_vector_tile, requests, json
from vt2geojson.tools import vt_bytes_to_geojson
from haversine import haversine
import datetime
import time

import pandas

import numpy as np
import Class_Image as Ci

from tensorflow.keras import Input, Model


async def get_image_at_url(path, coords):
    out = []
    for p in path:
        out.append(np.array(Ci.Image(p).preprocess_image(224)).reshape([1,224*224*3]))
    return np.array(out), coords

async def async_gen(gen):
    try:
        return next(gen)
    except StopIteration:
        return False, False


# Convert captured_at timestamp to a date
def msEpochToDate(in_ms_since_epoch):
    s = in_ms_since_epoch / 1000.0
    return(datetime.datetime.fromtimestamp(s).strftime('%Y-%m-%d %H:%M:%S.%f'))

class Mapillary:
    def __init__(self, access_token, search_b_left, search_t_right, zoom = 14, density=200, batch_size=16):
        self.density = density
        self.batch_size = batch_size
        self.search_b_left = search_b_left #Format [W, N]
        self.search_t_right = search_t_right  #Format [W, N]
        self.zoom = zoom
        self.reset()
        
        self.access_token = access_token
        
        # header for call to image endpoint
        self.headers= { "Authorization" : "OAuth {}".format(access_token) }
        
    def reset(self):
        self.tile = mercantile.tile(self.search_b_left[0], self.search_b_left[1], self.zoom) #tile that moves accross map
        
        self.b_left_tile = mercantile.tile(self.search_b_left[0], self.search_b_left[1], self.zoom) #tile at bottom left
        self.t_right_tile = mercantile.tile(self.search_t_right[0], self.search_t_right[1], self.zoom) #tile at top right
    
    def image_url_generator(self):

        # define an empty geojson as output
        output= { "type": "FeatureCollection", "features": [] }
        
        # loop through all tiles to get IDs of Mapillary data
        # Input coordinates would be the original point coordinates sent to the class
        
        filter_radius = 200
        
        # get the tiles with x and y coors intersecting bbox at zoom 14 only
        #tiles = list(mercantile.tiles(east, south, west, north, zoom))
        
        v_x = -1 if self.b_left_tile.x > self.t_right_tile.x else 1
        v_y = -1 if self.b_left_tile.y > self.t_right_tile.y else 1
        
        
        for x in range(self.b_left_tile.x, self.t_right_tile.x + v_x, v_x):
            for y in range(self.b_left_tile.y, self.t_right_tile.y + v_y, v_y):
                    tile = mercantile.tile(*mercantile.ul(x, y, self.zoom) + (self.zoom,))
                    #print(mercantile.bounds(x, y, self.zoom))
                   # print(mercantile.ul(tile.x, tile.y, tile.z), tile.x, tile.y)
                   # return
                
                    base = 'https://tiles.mapillary.com/maps/vtp/mly1_public/2'
                    tile_url = f'{base}/{tile.z}/{tile.x}/{tile.y}?access_token={self.access_token}'
                    #print(tile_url)
                    response = requests.get(tile_url)
                    if response.content.startswith(b"<!doctype html>"):
                        print("Mapillary API lockout, trying again in 600 seconds")
                        time.sleep(600)
                        y-= v_y
                        continue
                        
                    data = vt_bytes_to_geojson(response.content, tile.x, tile.y, tile.z)
                    
                    filtered_data = [feature for feature in data['features'] if feature['geometry']['type'] in ['Point']]
                    #TODO sort by date, get summer/winter
                    #summer May 15th - Oct 1st
                    #Winter December - March
                    
                    if(len(filtered_data) == 0):
                        #print("empty tile")
                        continue
                    
                    #Get lastest images
                    #filtered_data = sorted(filtered_data, key = lambda k: k['properties']['captured_at'], reverse=True) 
                    filtered_data = filtered_data[0:self.density]
                    
                    fields = ['captured_at','thumb_256_url','geometry']
                    fields = ','.join(fields)
                    
                    for d in filtered_data:
                        
                        image_endpoint_url = f'https://graph.mapillary.com/{d["properties"]["id"]}?fields={fields}&access_token={self.access_token}'
                        image_response = requests.get(image_endpoint_url, headers=self.headers).json()
                        #print(image_response)
                        #print(image_response['thumb_256_url'])
                        if 'thumb_256_url' not in image_response:
                            continue 
                            
                        yield image_response['thumb_256_url'], image_response['geometry']['coordinates']
                    

    def batched_image_generator(self):
        gen = self.image_url_generator()
        image_batch = []
        coords_batch = []
        for i, (url_path, coords) in enumerate(gen):
            image_batch.append(url_path)
            coords_batch.append(coords)
            if len(image_batch) == self.batch_size:
                yield np.array(image_batch), np.array(coords_batch)
                image_batch = []
                coords_batch = []

    async def generate_csv(self, model , save_path):
        model = Model(model.input, model.layers[24].output) #Trim to single half of ranking model
        model.training = False

        df = pandas.DataFrame(columns=[["long", "lat", "rank"]])
        gen = self.batched_image_generator()

        url_path, coords = next(gen)
        img_async = asyncio.create_task(get_image_at_url(url_path, coords))

        url_async = asyncio.create_task(async_gen(gen))

        i=0
        while True:
            i+=1
            
            url_path, coords = await url_async #next(gen)
            #url_path, coords = next(gen)
            if url_path is False: #on generator end
                break
                
            url_async = asyncio.create_task(async_gen(gen))

            
            img, coords_ = await img_async#get_image_at_url(url_path, coords)
            img_async = asyncio.create_task(get_image_at_url(url_path, coords))
            
            #build df long lat rank
            arr = np.concatenate((coords_, np.array(model([img[:,0], img[:,0]]))), 1)
            df =  df.append(pandas.DataFrame(arr, columns=df.columns), ignore_index=True)
            
            if (i * self.batch_size) % 500 == 0:
                df.to_csv(save_path, index=False)
            print(f"\r Processing image {i * self.batch_size} | {coords_[0][0]} {coords_[0][1]}", end="")
            
        df.to_csv(save_path, index=False)