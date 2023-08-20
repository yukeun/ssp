import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib as plt
import seaborn as sn
import sys
import pdb
# sys.path.insert(0, './g2p/g2p_en')
# from g2p.g2p_en.g2p import G2p


class LabelModel(nn.Module):
	'''
	Architecture:
		GRU network with 3 layers, 96 hidden dimensions
	'''
	def __init__(self, input_size=41, n_emb=500, n_class=40+1):
		# Label model을 통해서 출력되는 것은 41개의 class임.
		# 왜 41개냐? => 원래는 40개의 phoneme인데 ctc loss는 거기에 blank가 추가된 총 41개의
		# softmax output을 요구하기 때문
		# 즉 output은 41개의 phoneme에 대한 확률값들임을 암시한다.(=ctc input)
		# 그럼 ctc는 41개의 확률분포들을 가지고, 계산을 시작
		
		super(LabelModel, self).__init__()

		self.acoustic_encoder = GRU(input_size=input_size, 
                                    hidden_size=n_emb, 
                                    num_layers=3)
		self.emb_size = n_emb
		self.classifier = nn.Linear(self.emb_size, n_class)
		#40개의 class에서 +1 (blank)을 추가한 것을 확인하여 classify한다
		# class로 분류하기 위해서 다차원텐서를 1차원으로 linear하게 만들어준다
		self.log_softmax = nn.LogSoftmax(dim=-1)
		# log softmax를 이용하여 각 class 별 확률을 출력한다.


	def new_parameter(self, *size):
		out = nn.Parameter(torch.FloatTensor(*size))
		nn.init.xavier_normal_(out)
		return out

	def forward(self, acoustic_x, acoustic_x_length):
		batch_size = acoustic_x.size(0)

		# Extract acoustic embedding
		a_emb = self.acoustic_encoder(acoustic_x, acoustic_x_length)
		posterior_prob = self.log_softmax(self.classifier(a_emb))
		
		return posterior_prob 



class GRU(nn.Module):
	def __init__(self, input_size=41, hidden_size=500, 
              	num_layers=3, output_size=500, 
               	batch_first=True, bidirectional=True):
				
		super(GRU, self).__init__()
		
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.output_size = output_size
		self.batch_first = batch_first 
		self.bidirectional = bidirectional

		# model selection
		self.rnn = nn.LSTM(self.input_size, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, bidirectional=self.bidirectional)

		self.fc = nn.Linear(2 * self.hidden_size, self.output_size)
	
	def forward(self, x, x_length):

		batch_size = x.size(0)
		
		x = torch.nn.utils.rnn.pack_padded_sequence(x, x_length, batch_first=self.batch_first)

		out, (hn1) = self.rnn(x)
		
		#hidden_0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda()
		#out, (hn1) = self.rnn(x, (hidden_0))
		out, out_length = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=self.batch_first)
		emb = self.fc(out)

		return emb.permute(1, 0, 2)


	def get_embedding(self, x, x_length):
		return self.forward(x, x_length)


