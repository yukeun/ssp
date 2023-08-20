import os
import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from glob import glob
import torchaudio
from tqdm import tqdm
from utils.dataset import LibriSpeechDataset, Dataset, my_collate_fortriplet
from model.network import LabelModel

import pickle
from g2p.g2p_en import G2p
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

def load_audio(folder_path, path, feat_ext, minmax=0):
	# *** acoustic feature extraction ***
	#audio_path = '/hdd2/hkshin/server/Database/libriphraseSpeechCommands_v1/splitted_data/' + path
	audio_path = folder_path + path
	x, sr = torchaudio.load(audio_path, normalization=True)
	x = x.squeeze(0)

	if minmax == 1:
		x = minmax_scale(x, feature_range=(-0.99999, 0.99999))
		x = torch.from_numpy(x).float()
	# x = x.unsqueeze(0)
	x = x.cuda()
	feat = torch.transpose(feat_ext(x), 0, 1)
	feat = feat.unsqueeze(0)
	#feat = feat_ext(x)
	#feat = feat.squeeze(0)

	return feat

def beam_search(model, libri_rootpath, anchor_path, feat_ext, word, g2p, ctc_dict, phonemes, n_top_k, beam_size, minmax):
	del_cnt = 0
	seq_list = []
	# feat_ext : Mel or MFCC
	#===========================Enroll anchor===================
	acoustic_x = load_audio(libri_rootpath, anchor_path, feat_ext, minmax=minmax)	
	
	# Extract phoneme label
	anc_ctc_phn = g2p(word.replace('_', ' '))
	# print(anc_ctc_phn)
	anc_ctc_phn = [phn.replace('0', '').replace('1', '').replace('2', '') for phn in anc_ctc_phn]
	anc_ctc_label = map_phoneme_to_label(anc_ctc_phn, ctc_dict)
	anc_ctc_label = torch.LongTensor(anc_ctc_label)

	# print(word, anc_ctc_label)
	# print(bb)
	# Compute length of acoustic frames, phoneme sequence
	acoustic_length = [acoustic_x.size(1)]
	acoustic_length = torch.LongTensor(acoustic_length)

	text_length = [len(anc_ctc_label)]
	text_length = torch.LongTensor(text_length)
	if torch.cuda.is_available():
		anc_ctc_label = anc_ctc_label.cuda()

	# Compute log probability of phoneme sequence from the gru model
	posterior_prob = model(acoustic_x, acoustic_length)
	posterior_prob = posterior_prob.permute(1, 0, 2)
	cpu_pos_prob = posterior_prob[0].detach().cpu().numpy()
	argmax_seq = np.argmax(cpu_pos_prob, 1)
	phn_labels = [''] + phonemes

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
	print(beam_results)
	# beam_results - Shape: BATCHSIZE x N_BEAMS X N_TIMESTEPS 
	# A batch containing the series of characters 
	# (these are ints, you still need to decode them back to your text) 
	# representing results from a given beam search. 
	# Note that the beams are almost always shorter than the total number of timesteps, 
	# and the additional data is non-sensical, 
	# so to see the top beam (as int labels) from the first item in the batch, 
	# you need to run beam_results[0][0][:out_len[0][0]]



	return beam_results, beam_scores, timesteps, out_lens
	
# def eval_beam(test_filename, labelmodel, loss, feat_ext, device):
	# 1. 일단 파일에 있는 내용을 다 흡수
	 # beam_result가 batch contatining the series of characters
	 # timesteps - Shape: BATCHSIZE x N_BEAMS 
	 # The timestep at which the nth output character has peak probability. 
	 # Can be used as alignment between the audio and the transcript.
	 


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

class evaluate_beamserach():
	def __init__(self,file_path):
		self.file_path = file_path

