from prettytable import PrettyTable
import os

totals = []

def read():
	global totals
	with open('totals.txt') as f:
		totals = f.read().splitlines()
	totals = list(map(int, totals))
	f.close()

num_og_f_eng = 0
num_og_f_noneng = 1
num_aug_f_eng = 2
num_aug_f_noneng = 3

num_og_m_eng = 4
num_og_m_noneng = 5
num_aug_m_eng = 6
num_aug_m_noneng = 7

def rewrite():
	global totals
	with open('totals.txt', 'w') as f:
		for i in totals:
			f.writelines("%d\n" %i)
	f.close()

def og_f_eng():
	read()
	global num_og_f_eng
	global totals
	totals[num_og_f_eng] += 1
	rewrite()

def og_f_noneng():
	read()
	global num_og_f_noneng
	global totals
	totals[num_og_f_noneng] += 1
	rewrite()

def og_m_eng():
	read()
	global num_og_m_eng
	global totals
	totals[num_og_m_eng] += 1
	rewrite()

def og_m_noneng():
	read()
	global num_og_m_noneng
	global totals
	totals[num_og_m_noneng] += 1
	rewrite()

def aug_f_eng():
	read()
	global num_aug_f_eng
	global totals
	totals[num_aug_f_eng] += 1
	rewrite()

def aug_f_noneng():
	read()
	global num_aug_f_noneng
	global totals
	totals[num_aug_f_noneng] += 1
	rewrite()

def aug_m_eng():
	read()
	global num_aug_m_eng
	global totals
	totals[num_aug_m_eng] += 1
	rewrite()

def aug_m_noneng():
	read()
	global num_aug_m_noneng
	global totals
	totals[num_aug_m_noneng] +=1
	rewrite()

def main():
	os.system('clear')
	read()
	total_females = totals[num_og_f_eng] + totals[num_og_f_noneng]
	#Pretty table for female statistics
	print("Female")
	t = PrettyTable()
	t.field_names = ['*', 'Engineer', 'Non-engineer']
	t.add_row(['Original', totals[num_og_f_eng], totals[num_og_f_noneng]])
	t.add_row(['Synthetic', totals[num_aug_f_eng], totals[num_aug_f_noneng]])
	print(t)
	
	#Pretty table for male statistics
	total_males = totals[num_og_m_eng] + totals[num_og_m_noneng]
	print("Male")
	t = PrettyTable()
	t.field_names = ['*', 'Engineer', 'Non-engineer']
	t.add_row(['Original', totals[num_og_m_eng], totals[num_og_m_noneng]])
	t.add_row(['Synthetic', totals[num_aug_m_eng], totals[num_aug_m_noneng]])

	print(t)

main()
