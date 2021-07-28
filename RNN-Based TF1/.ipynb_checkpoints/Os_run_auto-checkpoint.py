import os
import sys

# ind_=[line for line in sys.stdin](0)
ind=int(sys.argv[1])
f=open('../phrase_generate/enron_sample_3words.txt','r')
msg_string_list= [line for line in f.readlines()]


# os.system('mkdir 2020-09-07T11-52-46/logs/stroke{}'.format(ind))
inc=3000

for i in range(ind*inc,(ind+1)*inc):
	string_=msg_string_list[i]
	string='a '+string_[:-1]+' a'
	print(string)
	tsteps=len(string)*12

	os.system("python run.py --sample --text "+string+" --tsteps {}".format(tsteps))
