import os
from glob import glob
import pandas as pd
from tqdm import tqdm
import pdb

# 걍 이해용 코드

def generate_label_dict():
	phonemes = ["<pad>", "<unk>", "<s>", "</s>"] \
		+ ['', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', \
		'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', \
		'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', \
		'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', \
		'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', \
		'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']

	# Make an empty dictionary for labeling phonemes.
	ctc_label = dict()
	# Map each phoneme to a label number.
	for i in range(len(phonemes)):
		ctc_label[phonemes[i]] = i + 1
	return ctc_label

def generate_int_label_dict():

	phonemes = ["<pad>", "<unk>", "<s>", "</s>"] \
		+ ['', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', \
		'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', \
		'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', \
		'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', \
		'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', \
		'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
	label_dict = dict()
	for i in range(len(phonemes)):
		label_dict[i+1] = phonemes[i]
	print(label_dict)
	return label_dict


def int_to_text(labels, label_dict):
	decoded = []
	for i in range(len(labels)):
		decoded.append(label_dict[labels[i]])
	return decoded


def map_phoneme_to_label(phoneme_seq, label_dict):
	label_list = []
	for i in range(len(phoneme_seq)):
		label = label_dict[phoneme_seq[i]]
		label_list.append(label)
	return label_list


def convert_top10k_phoneme():
	ctc_label = generate_label_dict()
	df = pd.read_csv('/home/hwayeon/Downloads/ctc_librispeech/data/librispeech_clean_train_100h_dataset.csv', na_filter=False)
	text_list = df['text']

	word_dict = dict()
	#for i in tqdm(range(len(text_list))):
	for i in tqdm(range(5)):
		pdb.set_trace()
		print(df['phoneme'][i][0])
		phoneme_list = str(df['phoneme'][i]).replace('[', '').replace(']', '').replace("'", "").replace(' ', '').split(',')
		print(phoneme_list)
		label_list = map_phoneme_to_label(phoneme_list, ctc_label)
		word_dict[text_list[i]] = label_list
	return word_dict
	

if __name__=='__main__':
	ctc_label = generate_label_dict()
	seq = ['AA0', 'UW', 'P', 'B']
	label_list = map_phoneme_to_label(seq, ctc_label)
	word_dict = convert_top10k_phoneme()
	print(word_dict)
	label_dict = generate_int_label_dict()
	decoded = int_to_text([3, 2, 5, 14], label_dict)
	print(decoded)
