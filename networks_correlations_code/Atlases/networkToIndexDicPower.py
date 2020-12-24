#!/usr/bin/python3
"""
==========================================================================================================================
Dictionary of all power networks. Key: network's name, Value: the indexes according to the order of the Power coordinates.
==========================================================================================================================
"""

import numpy as np

dic = {
"SSH" : range(0,31),
"SSM" : range(31,36),
"CO" : range(36,50),
"Auditory" : range(50,63),
"DMN" : range(63,121),
"Memory" : range(121,126),
"Visual" : range(126,157), 
"FP" : range(157,181), 
"Salience" : range(181,199),
"Subcortical" : range(199,212),
"VAN" : range(212,221),
"DAN" : range(221,232),
"Cerebellum" : range(232,236),
"Uncertain" : range(236,264)
}

labelToColorDic = {"Uncertain" : "olive", "SSH" : "cyan", "SSM" : "orange", "CO" : "purple", "Auditory" : "m", "DMN" : "red", "Memory" : "grey", 
	"Visual" : "blue", "FP" : "gold", "Salience" : "black", "Subcortical" : "brown", "VAN" : "teal", "DAN" : "green", "Cerebellum" : "purple"}