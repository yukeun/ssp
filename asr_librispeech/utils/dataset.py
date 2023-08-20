import pdb
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import splitext, basename, dirname
from torch.utils.data import Dataset
from tqdm import tqdm
import pdb
import torchaudio
from g2p.g2p_en import G2p


def plot_wav(x, fs, filename, normalize='normalization'):
	# wav파일을 plot하는 함수
	
	time = np.linspace(0, len(x)/fs, len(x))
	plt.plot(time, x)
	plt.xlabel('time (sec)')
	plt.xlim(0, time[-1])
	plt.ylim([-1, 1])
	plt.savefig('./picture/' + splitext(basename(filename))[0] + '_' + normalize + '.png')
	plt.clf()
	# wav파일을 plot하는 함수
 

def generate_label_dict():
	# label_dictionary 설정하는 함수

	phonemes = [' ', 'AA0', 'AA1', 'AA2', 'AE0', 'AE1', 'AE2', 'AH0', 'AH1', 'AH2', 'AO0', 
				'AO1', 'AO2', 'AW0', 'AW1', 'AW2', 'AY0', 'AY1', 'AY2', 'B', 'CH', 'D', 'DH', 
				'EH0', 'EH1', 'EH2', 'ER0', 'ER1', 'ER2', 'EY0', 'EY1', 'EY2', 'F', 'G', 'HH', 
				'IH0', 'IH1', 'IH2', 'IY0', 'IY1', 'IY2', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW0', 'OW1', 
				'OW2', 'OY0', 'OY1', 'OY2', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH0', 'UH1', 'UH2', 'UW', 
				'UW0', 'UW1', 'UW2', 'V', 'W', 'Y', 'Z', 'ZH']
	phonemes = sorted(list(set([phn.replace('0', '').replace('1', '').replace('2', '') for phn in phonemes])))
	# replace('a','b') = 'a'를 'b'로 대체
	# set : 중복을 제거
	# list : 리스트화
	# sorted : 기존의 리스트는 냅두고 새롭게 정렬하여 출력
	# 기존 phoneme은 총 61개지만, 묵음이라든지 비슷한 phoneme은 하나로 통합하여 size줄이자
	ctc_label = dict() # => 이거 알아보기 왜 ctc label이 필요할까
	for i in range(len(phonemes)):
		ctc_label[phonemes[i]] = i + 1

	return ctc_label


def map_phoneme_to_label(phoneme_seq, label_dict):
	# 여기서 phoneme_seq가 의미하는 바가 뭘까
	label_list = []

	for i in range(len(phoneme_seq)):
		phn_seq = phoneme_seq[i]
		phn_seq = phn_seq.replace('0', '').replace('1', '').replace('2', '')
		# label_dict에서와 마찬가지로 비슷한 phoneme끼리는 하나로 통합
		# 그래서 label이랑 매핑하는건가?
		label = label_dict[phn_seq]
		label_list.append(label)
  	
	return label_list


def convert_phoneme(df_word):
	ctc_label = generate_label_dict()
	print(ctc_label)
	g2p = G2p()
	word_dict = dict()
	word_dict['filename'] = []
	word_dict['label'] = []
	for i in tqdm(range(len(df_word['text']))):
			#tqdm은 진행률 바를 나타내려고 쓰는 것
		word_dict['filename'].append(df_word['audio'][i])
		
		sentence = df_word['text'][i]
		phoneme_seq = g2p(sentence)
		while(1):
			if "'" in phoneme_seq:
				idx = phoneme_seq.index("'")
				del phoneme_seq[idx]
				del phoneme_seq[idx-1]
			else:
				break
		phonemes = map_phoneme_to_label(phoneme_seq[0], ctc_label)
		word_dict['label'].append(phonemes)
	
	return word_dict


def my_collate_fortriplet(dataset):
	batch_size = len(dataset)
	data_acoustic = [dataset[i]['acoustic'] for i in range(batch_size)]
	data_text = [dataset[i]['text'] for i in range(batch_size)]
	align_leng = [data_acoustic[i].shape[1] for i in range(len(dataset))]
	text_leng = [len(data_text[i]) for i in range(len(dataset))]
	sorted_list = sorted(zip(align_leng, text_leng, data_acoustic, data_text), 
                      	 key=lambda t: t[0], reverse=True)
	len_feat, len_text, sorted_acoustic, sorted_text = zip(*sorted_list)
	longest_feat = max(len_feat)
	longest_text = max(len_text)
	frequency_axis = data_acoustic[0].shape[0]

	padded_feat = torch.zeros((batch_size, longest_feat, frequency_axis))
	padded_text = torch.zeros((batch_size, longest_text)).type(torch.LongTensor)

	for i in range(len(len_feat)):
		sequence = sorted_acoustic[i].transpose(1, 0)
		padded_feat[i, :len_feat[i], :] = sequence

	for i in range(len(len_text)):
		padded_text[i, :len_text[i]] = sorted_text[i]
	
	length_a = [item for item in len_feat] 
	length_t = [item for item in len_text] 

	return (padded_feat, padded_text), (length_a, length_t)


class LibriSpeechDataset(Dataset):
	def __init__(self, rootpath, filenames, feature_type='M', 
                 frm_length=400, hop_length=160, n_fft=512, 
                 sampling_rate=16000, win_type='hamming', 
                 n_mels=40, n_mfcc=13, device=None):
  				
		# set feature extraction parameters
		self.feature_type = feature_type
		self.frm_length = frm_length
		self.hop_length = hop_length
		self.n_fft = n_fft
		self.sampling_rate = sampling_rate
		self.win_type = win_type
		self.n_mels = n_mels
		self.n_mfcc = n_mfcc

		# Set data information.
		for k, filename in enumerate(filenames):
			# older_path = 'audio_data/train-clean-100'

			# Read train data.
			if "train" in filename:
				# folder_path = 'audio_data/train-clean-100' # 원본
				folder_path = 'audio_data/train-clean-100'
			elif "valid" in filename:
				folder_path = 'audio_data/dev-clean'
			self.df_word = pd.read_csv(filename, na_filter=False)
			for i in range(len(self.df_word)):
				spk, folder, _ = self.df_word['audio'][i].split('-')
				self.df_word['audio'][i] = os.path.join(folder_path, spk, folder, self.df_word['audio'][i])
			print(self.df_word['audio'][0])

		print(self.df_word)
		self.word_dict = convert_phoneme(self.df_word)
		print('len(self.df_word) = ', len(self.df_word))
		self.rootpath = rootpath
		print('self.rootpath = ', self.rootpath)

		self.melargs = dict()
		self.melargs['win_length'] = self.frm_length
		self.melargs['hop_length'] = self.hop_length
		self.melargs['n_fft'] = self.n_fft
		self.melargs['f_max'] = int(self.sampling_rate/2)
		self.melargs['n_mels'] = self.n_mels							
		self.C = torchaudio.transforms.MFCC(sample_rate=self.sampling_rate, 
                                      		n_mfcc=self.n_mfcc, dct_type=2, 
                                     		norm='ortho', melkwargs=self.melargs)
		self.M = torchaudio.transforms.MelSpectrogram(sample_rate=self.sampling_rate, 
                                            		  win_length=self.frm_length, 
                                                	  hop_length=self.hop_length, 
                                                   	  n_fft=self.n_fft, 
                                                      n_mels=self.n_mels)
		self.device = device
  
		if self.device:
			self.C = self.C.to(device)
			self.M = self.M.to(device)
	
	def __len__(self):
		# Return 100.
		return len(self.df_word)


	def __getitem__(self, index):
		acoustic = self.word_dict['filename'][index]
		acoustic = acoustic.replace('.wav', '.flac')
		label = self.word_dict['label'][index]
		# Extract acoustic features.
		x, sr = torchaudio.load(acoustic)
		x = x.to(self.device)
		#if word == '':
		#	word = 'sil_'	
		try:
			if self.feature_type == 'M':
				feat = self.M(x).to(self.device)
    
			elif self.feature_type == 'C':
				feat = self.C(x).to(self.device)
    
			feat = feat.squeeze(0)
			data = {'acoustic': feat, 'text': torch.LongTensor(label)}

			return data

		except:
			print('audio_len: ', x.shape[1])
			# Pad 0 if length is shorter than fft size.
			if x.shape[1] >= self.n_fft:
				print('x.shape = ', x.shape)

			x_tmp = torch.zeros((1, self.n_fft))
			x_tmp[:, :x.shape[1]] = x

			if self.feature_type == 'M':
				feat = self.M(x_tmp).to(self.device)
    
			elif self.feature_type == 'C':
				feat = self.C(x_tmp).to(self.device)
    
			feat = feat.squeeze(0)
			data = {'acoustic': feat, 'text': torch.LongTensor(label)}
			return data

			pass
		# *** sentence feature extraction ***
		#input_ids, token_type_ids, mask_ids = text_processing(text, self.tokenizer, self.max_input_length)

		# *** phoneme feature extraction ***
		# do at text network

if __name__=='__main__':
	ctc_label = generate_label_dict()
	print(ctc_label)
