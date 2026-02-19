#!/usr/bin/env python 
# Basic use of NASA API, example queries Astronomy picture of the day
from nasapy import Nasa
import os

try:
    with open('nasakey.txt', 'r') as f:
        key = f.read().strip()
except:
    print("sskey.txt not found")


nasa = Nasa(key)

# Astronomy Picture of the Day
apod = nasa.picture_of_the_day()
print(apod["title"], apod["url"])

