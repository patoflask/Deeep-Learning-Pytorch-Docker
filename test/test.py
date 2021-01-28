# -*- coding: utf-8 -*-
#!usr/bin/env python3
'''
=================================================================>--Version:v1
=========================================================>Compiègne:20/01/2021
=======================================================>Fraud predictive model
================================================>done by @Manfo Satana Patrice
'''

import requests

resp = requests.post("http://localhost:5001/predict", files={'file': open('2.jpeg', 'rb')},\
                     headers={'Cache-Control': 'no-cache'})

print(resp.text)