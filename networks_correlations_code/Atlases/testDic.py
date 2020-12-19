from networkToIndexDicGordon import dic
import pandas as pd
import argparse
from nilearn import plotting
import numpy as np

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--dic_excel', required=True, type=str, help='path to dictionary excel file')
	args = parser.parse_args()
	fd = pd.read_excel(args.dic_excel)
	for net, indexes in dic.items():
		fd_net	= fd[fd["Community"] == net]
		parcels = list(fd_net["ParcelID"])
		parcels = list(np.array(parcels) - np.array([1]*len(parcels)))
		if (indexes != parcels):
			print("Parcelation error in " , net)
		coords = fd_net["Centroid (MNI)"]
		coords_new = []
		for coord_str in coords:
			coords_new.append([float (x) for x in coord_str.split()])
		view = plotting.view_markers(coords_new, marker_size=10) 
		html_name = "test_out/network_plot" + str(net) + ".html"
		view.save_as_html(html_name)
		