import numpy as np
fp = open('training_embedding.txt')
fp_out = open('new_training_embedding.txt', 'w')
line = fp.readline()
while line:
	line = line.strip().split()
	fp_out.write('%s ' %line[0])
	list_index = []
	list_value = []
	for i in range(1, len(line)):
		id_val = line[i].split(':')
		list_index.append(int(id_val[0]))
		list_value.append(id_val[1])
	id_sorted = np.argsort(np.array(list_index))
	for i in id_sorted:
		fp_out.write('%d:%s ' %(list_index[i], list_value[i]))
	fp_out.write('\n')
	line = fp.readline()
fp.close()
fp_out.close()
