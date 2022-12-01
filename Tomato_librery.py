# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 18:10:55 2022

@author: eitan
"""




import pandas as pd
import numpy as np
from spectral import *
from scipy import stats

class Harvest():

    def __init__(self, tomato_shape, lab_results_path,reflectence_path):
      
      self.tomato_shape=tomato_shape
      self.lab_results=pd.read_csv(lab_results_path)
      self.cultivar,self.harvest_date=self._get_id()
      self.reflectence=pd.read_csv(reflectence_path)
      self.reflectence=self.reflectence.drop([self.reflectence.columns[0]],axis=1)
      self.reflectence=self.reflectence.T

    def _get_id(self):
      self.cultivar=self.lab_results.iloc[0][0].split('_')[0]
      self.harvest_date=self.lab_results.iloc[0][0].split('_')[1:4]
      return self.cultivar,self.harvest_date




class TomatoImage():
  
  def __init__(self,img_path, mask_path, bands_to_wave_table ,metadata=None,mask=None):
    self.img=self._load_img(img_path)#data type=
    self.mask=self.load_mask(mask_path)
    self.bands_to_waves= bands_to_wave_table
    self.metadata=metadata
  

  def spectral_to_RGB(self, img):
    R_wave=80 #red wavelength index in the 204 bands
    G_wave=45#red wavelength index in the 204 bands
    B_wave=20 # blue wavelength index in the 204 bands
    self.RGB=np.zeros([img.shape[0],img.shape[0],3],dtype=float)# create tamplate for RGB
    self.RGB[:,:,0]=(img[:,:,R_wave]-img[:,:,R_wave].min())/(img[:,:,R_wave].max()-img[:,:,R_wave].min())#red 
    self.RGB[:,:,1]=(img[:,:,G_wave]-img[:,:,G_wave].min())/(img[:,:,G_wave].max()-img[:,:,G_wave].min())#green
    self.RGB[:,:,2]=(img[:,:,B_wave]-img[:,:,B_wave].min())/(img[:,:,B_wave].max()-img[:,:,B_wave].min())#blue
    #save as jpeg
    return self.RGB
  
  def _load_img(self,img_path):
    img = open_image(img_path)
    self.spectral=np.array(img.load())
    self.spectral=np.rot90(self.spectral[:,:,:],3)#rotate 
    self.RGB = self.spectral_to_RGB(self.spectral)
    return self.spectral

  def load_mask(self, mask_path):
    pass
  
class Tomato(Harvest):
    def __init__(self,lab_results,harvest:Harvest,tomato_id:int, reflectence,tomato_img=None):
      self.reflectence=reflectence
      self.lab_results=lab_results
      self.tomato_id=self.lab_results.iloc[0]
      self.harvest_date=harvest.harvest_date
      self.cultivar=harvest.cultivar
      if tomato_img==None:
        pass
      self.tomato_img=tomato_img
    def update_ref(self, reflectence):
      pass


class TomatoList():

  def __init__(self,harvest_list,tomato_img=None ,poligon_file=None ,):
    self.tomato_img=tomato_img
    self.poligon_file=poligon_file
    self.tomato_dict=self._harvest_list_to_tomato_dict(harvest_list)
    self.tomato_list=self.creat_tomato_list(self.tomato_dict)
  def creat_tomato_list(self,tomato_dict):
    return [tomato_dict[i] for i in range(1,len(tomato_dict.keys())+1)]
  def _harvest_list_to_tomato_dict(self,harvest_list:list)->dict:
    tomato_dict={}
    key_count=1
    for i in harvest_list:
      temp=create_Tomato_instances(i)
      for j in range (key_count,(key_count+len(temp.keys()))):
        tomato_dict[j]=temp[j-key_count]
      key_count+=len(temp.keys())
    return tomato_dict
      
class TomatoFromImage():
    def __init__(self, harvest:Harvest,tomato_id, tomato_imgs, lab_results_path:str, reflectence:pd.DataFrame=None):
      self.reflectence=reflectence
      self.lab_results=pd.read_csv(self.lab_results_path)
      self.tomato_imgs=tomato_imgs
      self.harvest=harvest

      self.tomato_id=tomato_id

    def update_ref(self, reflectence):
      pass
def calc_stats_per_band(tomato_img):
  np_image_array = tomato_img.img
  band_depth=np_image_array.shape[2]
  stats={}
  mask=tomato_img.mask.flatten()
  for i in range(band_depth):
    flat=np_image_array[:,:,i].flatten()
    filterd = flat * mask
    filterd[mask==0]=None
    stats[i]= (flat.min(), flat.max() ,flat.mean() , np.median(flat) ,flat.std())
  return stats

def create_Tomato_instances(harvest:Harvest)->dict:
  '''function that gest a harvest
   object and creates tomato instences 
   in a dictionery with the key value the id'''
  T={}
  for i in range(harvest.lab_results.shape[0]):
    T[i]=Tomato(harvest.lab_results.iloc[i],harvest,i, harvest.reflectence.iloc[i])
  return T

def NI_model(tomato_list):
  ''' input TomatoList obj, 
  returns r_2,rmse color maps and b1,b2 of highest R2'''
  band_depth=204
  r_2 = np.arange(band_depth*band_depth).reshape(band_depth, band_depth)
  r_2=np.float16(r_2)
  rmse =r_2.copy()
  np_results=np.array([tomato_list.tomato_list[i].lab_results.iloc[1:] for i in range(0,len(tomato_list.tomato_list))])
  df_r=np.array([tomato_list.tomato_list[i].reflectence for i in range(0,len(tomato_list.tomato_list))])
  r_2_max,b1,b2=0,0,0
  for j in range(band_depth):#running all combinations
    for y in range (band_depth):
            # np_results[:,np_results != np.array(None)]
            n=(df_r[:,y]-df_r[:,j])/(df_r[:,y]+df_r[:,j])
            slope, intercept, r_value, p_value, std_err = stats.linregress(n,np_results) 
            r_2[y,j] =  r_value**2
            rmse[y,j] =np.sqrt(((np_results-(intercept + (slope * n)))**2).mean())
  return r_2,rmse