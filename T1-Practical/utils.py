import scipy.cluster.vq as sp
import numpy as np
import os, pickle
from myhmm_scaled import MyHmmScaled as HMM

def obtain_training_data(trng_path="./training"):
	trng_data = {}
	vs = {}								# Vector Sizes Mapping
	dis_files = os.listdir(trng_path)
	for df in dis_files:
		disease = df.split(".")[0]
		trng_data[disease] = []
		f = open(trng_path+'/'+df,"r")
		l = f.readlines()
		f.close()
		for i in range(len(l)):
			s1 = l[i][:-2]
			s2 = s1.replace("null","0")
			trng_data[disease].append(map(float, s2.split(",")[1:]))
		d_len = len(trng_data[disease][0])
		if not vs.has_key(d_len):
			vs[d_len]= []
		vs[d_len].append(disease)
	pickle.dump(vs, open("size_mapping.pkl","wb"))
	return trng_data, vs

def vector_quantize(data_dict, vs, bins):
	codebooks = {}
	vq_data = {}
	for size in vs.keys():
		all_size_data = []
		for disease in vs[size]:
			all_size_data.extend(data_dict[disease])
		codebooks[size] = sp.kmeans(np.asarray(all_size_data), bins)[0]
	pickle.dump(codebooks,open("all_codebooks.pkl","wb"))
	for dis in data_dict.keys():
		n = len(data_dict[dis])
		m = len(data_dict[dis][0])
		vq_data[dis] = map(str,sp.vq(np.reshape(data_dict[dis],(n,m)), codebooks[len(data_dict[dis][0])])[0])
	return vq_data

if __name__=="__main__":
	bins = 16
	trng, vec_sizes = obtain_training_data()
	vq_data = vector_quantize(trng, vec_sizes, bins)
