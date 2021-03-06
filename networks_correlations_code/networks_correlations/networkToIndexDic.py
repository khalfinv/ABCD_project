#!/usr/bin/python3
"""
==========================================================================================================================
Dictionary of all Gordon networks. Key: network's name, Value: the indexes according to the order of the Gordon coordinates.
==========================================================================================================================
"""

dic = {
"SMhand" : [1,29,30,31,32,34,35,36,37,44,45,46,47,49,53,55,56,57,162,189,190,192,193,194,200,201,203,204,205,206,208,209,212,213,214,215,216,269],
"SMmouth" : [2,38,52,58,163,196,211,217],
"CinguloParietal" : [11,88,92,172,253],
"CinguloOperc" : [20,21,26,27,33,39,62,70,71,75,80,81,83,100,102,104,110,111,146,152,179,180,184,186,187,191,195,197,218,222,233,234,237,244,245,247,248,273,316,317],
"Auditory" : [9,63,64,65,66,67,68,69,76,101,103,159,170,223,226,229,231,232,238,243,267,268,328,329],
"Default" : [0,3,5,24,25,43,93,113,115,116,125,126,144,145,149,150,151,153,155,156,161,164,183,185,199,219,224,256,258,277,278,289,314,315,320,321,322,323,324,325,330],
"Visual" : [4,7,14,15,16,19,89,96,97,98,130,131,135,136,137,138,139,140,165,168,174,175,176,250,254,255,257,262,263,264,266,292,297,298,306,307,308,309,310], 
"FrontoParietal" : [6,8,23,77,95,107,108,147,148,166,167,169,181,239,259,260,271,272,275,276,318,319,326,327], 
"Salience" : [28,82,182,246],
"VentralAttn" : [22,59,60,61,74,78,79,84,85,157,160,220,221,225,227,228,230,236,240,241,242,331,332],
"DorsalAttn" : [40,41,42,48,50,51,54,73,86,87,90,91,94,99,105,106,109,112,154,188,198,202,207,210,235,249,251,252,261,265,270,274],
"None" : [10,17,18,72,114,117,118,119,120,121,122,123,124,127,128,132,133,134,141,143,158,171,177,178,279,280,281,282,283,284,285,286,287,288,290,291,295,296,299,300,301,302,303,304,305,311,313],
"RetrosplenialTemporal" : [12,13,129,142,173,293,294,312],
#Cannot find the coordinates for those subcortical regions
# "CEREBELLUM" : [333,343],
# "THALAMUS" : [334,344],
# "CAUDATE" : [335,345],
# "PUTAMEN" : [336, 346],
# "PALLIDUM" : [337, 347],
# "BRAIN_STEM" : [338],
# "HIPPOCAMPUS" : [339,348],
# "AMYGDALA" : [340,349],
# "ACCUMBENS" : [341, 350],
# "DIENCEPHALON_VENTRAL" : [342,351]
}

labelToColorDic = {"None" : "olive", "SMhand" : "cyan", "SMmouth" : "orange", "CinguloOperc" : "purple", "Auditory" : "m", "Default" : "red", "RetrosplenialTemporal" : "grey", 
	"Visual" : "blue", "FrontoParietal" : "gold", "Salience" : "black", "CinguloParietal" : "brown", "VentralAttn" : "teal", "DorsalAttn" : "green", "CEREBELLUM" : "royalblue", "THALAMUS" : "lime", 
	"CAUDATE" : "coral", "PUTAMEN" : "crimson", "PALLIDUM" : "slateblue", "BRAIN_STEM" : "khaki", "HIPPOCAMPUS" : "seagreen", "AMYGDALA" : "navy", "ACCUMBENS" : "hotpink", "DIENCEPHALON_VENTRAL" : "yellow"}