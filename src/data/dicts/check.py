fp = open('id_full.txt')
line = fp.readline()
i = 0
while line:
	i += 1
	line = line.strip().split()
	if len(line) > 2:
		print line
	if len(line[0]) < 2:
		print i, line[0]
	if i==198517:
		print line[0]
	line = fp.readline()
