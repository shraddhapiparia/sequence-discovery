from gensim.models import Word2Vec
import collections
import operator
from collections import defaultdict
import numpy as np
import os


datapath = "train_data"
all_user_files = sorted(os.listdir(datapath))

length = [3,4,6,8]


for seq_len in length:
    train_subseq_dict={}
    test_subseq_dict={}
    for user_file in all_user_files:
	file_path = os.path.join(datapath , user_file)

	with open(file_path,'r') as data_file:
	    train_data = data_file.readlines()
	    #split = float(0.7) * len(data_elements)
	    #print split, type(split)
	    
	    #train_data = data_elements[0:int(split)]
	    for line_idx in range(0,len(train_data)):
		end_idx = line_idx + seq_len
		if end_idx < len(train_data):
		    subseq = train_data[line_idx:end_idx]
		    subseq = subseq[seq_len-1].strip("\n")
		    if subseq in train_subseq_dict:
			train_subseq_dict[subseq] += 1
		    else:
			train_subseq_dict[subseq] = 1

    

    res_file = "train_with_length_"+str(seq_len-1)+".txt"
    with open(res_file,'w') as result_file:		    
	for key,val in sorted(train_subseq_dict.items(), reverse=True, key=lambda item : item[1]):
	    result_file.write(str(key))
	    result_file.write(" ")
	    result_file.write(str(val))
	    result_file.write("\n")

'''
f = open('input_data.txt',"r")
while True:
    if not nextval:
	line1 = f.readline()
    else: 
	line1 = nextval
    line2 = f.readline()
    if not line2: 
	break
    else:
	nextval = f.readline()
	sentences.append(line1+line2+nextval)
	classes[nextval] += 1
	#nextval = f.readline()
	#sentences.append(line2)
for value,count in sorted(classes.items(), key=operator.itemgetter(1) , reverse = True):
	print value,count

for value in sentences:
	print value

input_text = f.read()

for line in input_text.split('\n'):
    action = line
	
    sentences.append([action])
    vocab.append(action)'''
