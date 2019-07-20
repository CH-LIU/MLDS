import numpy as np

def make_tag(filename):
	tag_list = []
	hair_dic = {}
	eye_dic = {}
	hair_count =0
	eye_count = 0
	with open(filename, 'r') as tag_file:
		for tag in tag_file.readlines():
			tag = tag.replace('\n','')
			t = tag.split(' ')
			tt =t[0].split(',')
			hair = tt[1] + " hair"
			eye = t[2]+" eye"
			if hair not in hair_dic : 
				hair_dic[hair] = hair_count
				hair_count+=1
			if eye not in eye_dic : 
				eye_dic[eye] = eye_count
				eye_count+=1
	l = len(hair_dic)
	with open(filename, 'r') as tag_file:
		for tag in tag_file.readlines():
			temp = [0 for i in range(22)]
			tag = tag.replace('\n','')
			t = tag.split(' ')
			tt =t[0].split(',')
			hair = tt[1]+ " hair"
			eye = t[2] + " eye"
			temp[hair_dic[hair]] += 1
			temp[eye_dic[eye]+12] += 1
			tag_list.append(temp)
	tag_list = np.array(tag_list)
	return tag_list, hair_dic, eye_dic
