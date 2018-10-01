op_file=open('hi-ud-train_2.conllu','w')
with open('hi-ud-train.conllu','r') as ip_file:
	for line in ip_file:
		line=line.split()
		if len(line)==10:
			line[5]+='|'+line[9]
		op_file.write('\t'.join(line)+'\n')
op_file.close()


op_file=open('hi-ud-test_2.conllu','w')
with open('hi-ud-test.conllu','r') as ip_file:
	for line in ip_file:
		line=line.split()
		if len(line)==10:
			line[5]+='|'+line[9]
		op_file.write('\t'.join(line)+'\n')
op_file.close()