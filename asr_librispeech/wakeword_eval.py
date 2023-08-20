import pdb
import time
import os
import datetime
import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
# custom package
#from network import visualization_affinity_mat
import pickle
# from g2p.g2p_en import G2p
from g2p_en import G2p
from ctcdecode import CTCBeamDecoder
from sklearn.metrics import roc_curve, roc_auc_score

def generate_label_dict():
	phonemes = [' ', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', \
			'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', \
			'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', \
			'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', \
			'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', \
			'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
	phonemes = sorted(list(set([phn.replace('0', '').replace('1', '').replace('2', '') for phn in phonemes])))

	ctc_label = dict()
	for i in range(len(phonemes)):
			ctc_label[phonemes[i].replace('0', '').replace('1', '').replace('2', '')] = i + 1
#			   print(ctc_label)
	return ctc_label


def map_phoneme_to_label(phoneme_seq, label_dict):
	label_list = []
	for i in range(len(phoneme_seq)):
		phn_seq = phoneme_seq[i].replace('0', '').replace('1', '').replace('2', '').replace("'", ' ')
		# phn_seq = phoneme_seq[i].replace("'", ' ')
		# if phoneme_seq[i][-1] in ['0', '1', '2']:
		# 	phn_seq = phn_seq.replace('0', '').replace('1', '').replace('2', '')
			# if i > 0 and phn_seq == phoneme_seq[i-1].replace('0', '').replace('1', '').replace('2', '').replace("'", ' '):
			# 	continue
		label = label_dict[phn_seq]
		label_list.append(label)
#			   print(label_list)
	return label_list


def convert_phoneme(df_word):
	ctc_label = generate_label_dict()
	g2p = G2p()
	word_dict = dict()
	word_dict['filename'] = []
	word_dict['label'] = []
	for i in tqdm(range(len(df_word['text']))):
		#word_dict['filename'].append(df_word['audio_filename'][i])
		word_dict['filename'].append(df_word['audio'][i])
		#spk, folder, _ = df_word['audio'][i].split('-')
		#word_dict['filename'].append(os.path.join(spk, folder, df_word['audio'][i]))

		sentence = df_word['text'][i]
		phoneme_seq = g2p(sentence)
		if "'" in phoneme_seq:
			#print(phoneme_seq)
			idx = phoneme_seq.index("'")
			del phoneme_seq[idx]
			#phoneme_seq.del(idx+1)
			del phoneme_seq[idx-1]
		#if "'" in phoneme_seq:
		#	   print(sentence)

		#print(sentence)
		phonemes = map_phoneme_to_label(phoneme_seq, ctc_label)
		#phonemes = map_phoneme_to_label(g2p(sentence), ctc_label)
		word_dict['label'].append(phonemes)
	return word_dict

import sklearn
import torchaudio
from sklearn.preprocessing import minmax_scale
from trainer import per

def load_audio(folder_path, path, feat_ext, minmax=0):
	# feature를 extract하면 굳이 넣어줄 필요 없지 않나?
	# *** acoustic feature extraction ***
	#audio_path = '/hdd2/hkshin/server/Database/libriphraseSpeechCommands_v1/splitted_data/' + path
	
	#folder_path : libri audio가 있는 경로
	#path : filename


	spk, folder, _ = path.split('-')
		# '-'을 기준으로 'audio'클래스를 쪼개서 출력
		# ex) 출력값 => spk : 3214, folder : 167607 
	path1 = os.path.join(folder_path, spk, folder, path)
	audio_path = path1
	x, sr = torchaudio.load(audio_path)
	# x, sr = torchaudio.load(audio_path, normalization=True)
	x = x.squeeze(0)

	if minmax == 1:
		# x = minmax_scale(x, feature_range=(-0.99999, 0.99999))
		x = torch.from_numpy(x).float()
	# x = x.unsqueeze(0)
	x = x.cuda()
	feat = torch.transpose(feat_ext(x), 0, 1)
	feat = feat.unsqueeze(0)
	#feat = feat_ext(x)
	#feat = feat.squeeze(0)

	return feat

def enroll_donut(model, libri_rootpath, anchor_path, feat_ext, phonemes, n_top_k=20, beam_size=10):
	#def enroll_donut(model, libri_rootpath, anchor_path, feat_ext, word, g2p, ctc_dict, phonemes, n_top_k, beam_size, minmax)
	
	del_cnt = 0
	seq_list = []
	#===========================Enroll anchor===================
	acoustic_x = load_audio(libri_rootpath, anchor_path, feat_ext)
	# acoustic_x = load_audio(libri_rootpath, anchor_path, feat_ext, minmax=minmax)	#원본
	# libri_rootpath : libri audio가 있는 경로
	# anchor_path : 구체적인 경로(파일 이름이 되겠네) = filename

	# # Extract phoneme label
	# anc_ctc_phn = g2p(word.replace('_', ' '))
	# # print(anc_ctc_phn)
	# anc_ctc_phn = [phn.replace('0', '').replace('1', '').replace('2', '') for phn in anc_ctc_phn]
	# anc_ctc_label = map_phoneme_to_label(anc_ctc_phn, ctc_dict)
	# anc_ctc_label = torch.LongTensor(anc_ctc_label)

	# print(word, anc_ctc_label)
	# print(bb)
	# Compute length of acoustic frames, phoneme sequence
	acoustic_length = [acoustic_x.size(1)]
	acoustic_length = torch.LongTensor(acoustic_length)

	# text_length = [len(anc_ctc_label)]
	# text_length = torch.LongTensor(text_length)
	#if torch.cuda.is_available():
	#	anc_ctc_label = anc_ctc_label.cuda()

	# Compute log probability of phoneme sequence from the gru model
	posterior_prob = model(acoustic_x, acoustic_length)
	posterior_prob = posterior_prob.permute(1, 0, 2)
	cpu_pos_prob = posterior_prob[0].detach().cpu().numpy()
	argmax_seq = np.argmax(cpu_pos_prob, 1)
	# pdb.set_trace()
	phn_labels = [''] + phonemes # blank를 첨가

	# Beam search decoding
	decoder = CTCBeamDecoder(
		phn_labels,
		model_path=None,
		alpha=0,
		beta=0,
		cutoff_top_n=n_top_k,
		cutoff_prob=1.0,
		beam_width=beam_size,
		num_processes=4,
		blank_id=0,
		#log_probs_input=False
		log_probs_input=True
	)
	#beam_results, beam_scores, timesteps, out_lens = decoder.decode(output)
	beam_results, beam_scores, timesteps, out_lens = decoder.decode(posterior_prob)
	return beam_results, beam_scores, timesteps, out_lens

def beam_evaluation(libri_test_filename, libri_rootpath, labelmodel, loss, feat_ext):
	# phonemes = generate_int_label_dict()
	df_word = pd.read_csv('/home/dbrms7459/ctc_librispeech/data/test_2h.csv', na_filter=False)
	phonemes = [' ', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', \
			'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', \
			'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', \
			'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', \
			'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', \
			'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
	phonemes = sorted(list(set([phn.replace('0', '').replace('1', '').replace('2', '') for phn in phonemes])))
	# 이때 phonemes는 dict이 아닌 
	phn_labels = [''] + phonemes 
	g2p = G2p()
	count = len(df_word['text'])
	per_sum = 0
	for i in tqdm(range(len(df_word['text']))):
		filename = df_word['audio'][i]
		text = df_word['text'][i] # text 출력 => 얘를 phoneme으로 바꿔줘야한다
		# print(text)
		
		sentence = df_word['text'][i]
		phoneme_seq = g2p(sentence)
		# print(phoneme_seq)
		# pdb.set_trace()
		if "'" in phoneme_seq:
			#print(phoneme_seq)
			idx = phoneme_seq.index("'")
			del phoneme_seq[idx]
			#phoneme_seq.del(idx+1)
			del phoneme_seq[idx-1]

		for i in range(len(phoneme_seq)):
			phoneme_seq[i] = phoneme_seq[i].replace('0', '').replace('1', '').replace('2', '').replace("'", ' ')
		# print(phoneme_seq)
		# pdb.set_trace()
		beam_results, beam_scores, timesteps, out_lens = enroll_donut(labelmodel, libri_rootpath, filename, feat_ext, phonemes)
		output_str = convert_to_string(beam_results[0][0], phn_labels, out_lens[0][0])
		# print(output_str)
		# print(per(phoneme_seq, output_str) * 100)
		per_sum += (per(phoneme_seq,output_str) * 100)
		# pdb.set_trace()
	avg_per = (per_sum / count)
	print(avg_per)


def convert_to_string(token, labels, seq_len):
	decoded_seq = [labels[x] for x in token[0:seq_len]]
	return decoded_seq


def generate_int_label_dict():

	#phonemes = ["<pad>", "<unk>", "<s>", "</s>"] \
	#		+ ['', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', \
	#		'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', \
	#		'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', \
	#		'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', \
	#		'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', \
	#		'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
	phonemes = [' ', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', \
			'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', \
			'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', \
			'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', \
			'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', \
			'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
	phonemes = sorted(list(set([phn.replace('0', '').replace('1', '').replace('2', '') for phn in phonemes])))
	label_dict = dict()
	for i in range(len(phonemes)):
			label_dict[i+1] = phonemes[i]
#	print(label_dict)
	return label_dict
