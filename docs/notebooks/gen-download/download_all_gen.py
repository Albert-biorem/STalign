# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 08:18:19 2024

@author: Albert
"""
import time
import os
import patoolib
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options 
import csv
def check_exists_by_xpath(drivr,xpath):
    try:
        drivr.find_element(By.XPATH,xpath)
    except NoSuchElementException:
        return False
    return True
#mirar quin es l'ultim document generat en una carpeta
def latest_download_file(path):
      newest=""
      os.chdir(path)
      files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
      if files:
          newest = files[-1]

      return newest

cur_dir=os.getcwd()
option = webdriver.ChromeOptions()
with open('query.csv', newline='') as f:
    reader = csv.reader(f)
    data = list(reader)
url="http://api.brain-map.org/grid_data/download/"
for i in data:
    if i[3]=="2":
        newpath = cur_dir+"\\gen_list\\"+i[1]+"\\"
        if not os.path.exists(newpath):
           os.makedirs(newpath)
        os.chdir(newpath)
        prefs = {"download.default_directory": newpath}
        #example: prefs = {"download.default_directory" : "C:\Tutorial\down"};
        option.add_experimental_option("prefs", prefs)
    
        driver = webdriver.Chrome(options=option)
        a=url+i[2]
        driver.get(a)
        fileends = "crdownload"
        while "crdownload" == fileends:
            time.sleep(2)
            newest_file = latest_download_file(newpath)
            fileends=newest_file.split(".")[-1]
        try:
            patoolib.extract_archive(newest_file, outdir=newpath)
            os.remove(newest_file)
        except:
             print("Can't not unzip "+newest_file)
        os.chdir(cur_dir)     