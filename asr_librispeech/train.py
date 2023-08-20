import os
import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
# custom package
from utils.dataset import LibriSpeechDataset, my_collate_fortriplet
from model.network import LabelModel
from trainer import fit, evaluate
import pdb


def get_parser():
	# 인자값을 받을 수 있는 인스턴스 생성
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)

	# 입력받을 인자값 등록
	parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3])
	parser.add_argument('--epoch', type=int, default=100) # 200
	parser.add_argument('--batchsize', type=int, default=32) # 32

	# parser.add_argument('--dataset', type=str, default='librisentence', choices=['librisentence'])

	parser.add_argument('--feature', type=str, default='M', choices=['M', 'C'])
	parser.add_argument('--sr', type=int, default=16000)
	parser.add_argument('--nfft', type=int, default=512) # 512
	parser.add_argument('--wintype', type=str, default='hanning', choices=['hamming', 'hanning'])
	parser.add_argument('--nmels', type=int, default=41) # 41
	parser.add_argument('--nmfcc', type=int, default=13)
	parser.add_argument('--checkpoint', type=str, default="")
	return parser


if __name__ == '__main__':

	
	# Set dataset parameters.
	# 입력받은 인자값을 args에 저장 (type: namespace)
	args = get_parser().parse_args()
	
	gpu_number = args.gpu
	batch_size = args.batchsize
	feature_type = args.feature
	sampling_rate = args.sr
	frm_length = int(np.floor(0.025 * sampling_rate))
	hop_length = int(np.floor(0.010 * sampling_rate))
	n_fft = args.nfft
	win_type = args.wintype
	n_mels = args.nmels
	n_mfcc = args.nmfcc

	# datasetname = args.dataset

	# Set GPU device.
	device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
	print('device = ', device)

	# Set input feature.
	# parser.add_argument('--feature', type=str, default='M', choices=['M', 'C'])에서 feature 선택
	# M이면 mel, C이면 mfcc
	if feature_type == 'M':
		input_size = n_mels
	elif feature_type == 'C':
		input_size = n_mfcc

	# Load dataset.		
	libri_rootpath = 'audio_data/train-clean-100/'

	libri_train_filename = ['/home/dbrms7459/ctc_librispeech/data/train_15h.csv']
	libri_valid_filename = ['/home/dbrms7459/ctc_librispeech/data/valid_2h.csv']

	train_dataset = LibriSpeechDataset(libri_rootpath, 
                                       libri_train_filename, 
                                       feature_type,
									   frm_length, hop_length, 
            						   n_fft, sampling_rate, 
                     				   win_type, n_mels, 
                            		   n_mfcc, device)
	# => /home/dbrms7459/ctc_librispeech/utils/dataset.py에 해당 데이터셋 설정해놈

	
	val_dataset = LibriSpeechDataset(libri_rootpath, 
                                  	 libri_valid_filename, 
                                     feature_type, 
                                     frm_length, hop_length, 
                                     n_fft, sampling_rate, 
                                     win_type, n_mels, 
                                     n_mfcc, device)
	# => /home/dbrms7459/ctc_librispeech/utils/dataset.py에 해당 데이터셋 설정해놈
	my_collate = my_collate_fortriplet

# 데이터 설정 후에는 항상 dataloader에서 batch 및 기타 하이퍼파라미터 설정하기
	train_loader = torch.utils.data.DataLoader(train_dataset, 
                                               batch_size=batch_size, 
                                               shuffle=True, 
                                               collate_fn=my_collate, 
                                               num_workers=0,  # cpu core setting 오류 없이 할려면 0으로 설정
                                               pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False,
                                             collate_fn=my_collate, 
                                             num_workers=0, 
                                             pin_memory=True)
  	# Set model and training parameters
	labelmodel = LabelModel(input_size)

	print('Model params. : ', 
       	  sum([param.nelement() for param in labelmodel.parameters()]))
 
	# Make folder to save results and trained model.
	if 'result' not in os.listdir():
		os.mkdir('./result')
	if 'checkpoints' not in os.listdir():
		os.mkdir('./checkpoints')

	# Set optimizer.
	lr = 1e-4
	optimizer = optim.Adam(labelmodel.parameters(), lr=lr, 
                           betas=(0.9, 0.99), eps=1e-12)
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
											   mode='min', 
											   factor=0.9, 
										       patience=3, 
											   threshold=0.0001,
											   threshold_mode='rel', 
											   cooldown=0, 
											   min_lr=0, 
											   eps=1e-08, 
											   verbose=False)

	# Set loss function.
	loss = torch.nn.CTCLoss()

	# Load checkpoint.
	if os.path.exists(args.checkpoint):
		checkpoint = torch.load(args.checkpoint)
		labelmodel.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		last_epoch = checkpoint['epoch']
	else:
		last_epoch = 0

	if torch.cuda.is_available():
		labelmodel = labelmodel.to(device)
		loss = loss.to(device)

	n_epochs = args.epoch

	# Train model. 
	print('--------------- start training ----------------')
	fit(train_loader, val_loader, labelmodel, loss, optimizer, 
     	scheduler, batch_size, n_epochs, device, total_loss=None, 
    	last_epoch=last_epoch)
	print('---------------- end training -----------------')

	# Evaluate model.
	print('------------finish------------')
