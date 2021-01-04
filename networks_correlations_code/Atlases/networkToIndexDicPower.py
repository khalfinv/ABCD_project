#!/usr/bin/python3
"""
==========================================================================================================================
Dictionary of all power networks. Key: network's name, Value: the indexes according to the order of the Power coordinates.
==========================================================================================================================
"""

import numpy as np

dic = {
"SSH" : range(0,30),
"SSM" : range(30,35),
"CO" : range(35,49),
"Auditory" : range(49,62),
"DMN" : range(62,120),
"Memory" : range(120,125),
"Visual" : range(125,156), 
"FP" : range(156,181), 
"Salience" : range(181,199),
"Subcortical" : range(199,212),
"VAN" : range(212,221),
"DAN" : range(221,232),
"Cerebellum" : range(232,236),
"Uncertain" : range(236,264)
}

labelToColorDic = {"Uncertain" : "olive", "SSH" : "cyan", "SSM" : "orange", "CO" : "purple", "Auditory" : "m", "DMN" : "red", "Memory" : "grey", 
	"Visual" : "blue", "FP" : "gold", "Salience" : "black", "Subcortical" : "brown", "VAN" : "teal", "DAN" : "green", "Cerebellum" : "purple"}