import os
import argparse
import torch
import numpy as np
import torch.optim as optim
from torch.autograd import Variable
from glob import glob
import torchaudio
from utils.dataset import LibriSpeechDataset, my_collate_fortriplet
from model.network import LabelModel


def get_parser():
	# 기본 하이퍼파라미터 설정
	parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
	parser.add_argument('--gpu', type=int, default=0, choices=[0, 1, 2, 3])
	parser.add_argument('--batchsize', type=int, default=32) # 32

	parser.add_argument('--dataset', type=str, default='libri', choices=['libri'])
	parser.add_argument('--feature', type=str, default='M', choices=['M', 'C'])
	parser.add_argument('--n_enroll', type=int, default=3)
	parser.add_argument('--sr', type=int, default=16000) # sampling rate 16000Hz
	parser.add_argument('--nfft', type=int, default=512) # fft size : 512
	parser.add_argument('--wintype', type=str, default='hanning', choices=['hamming', 'hanning'])
	parser.add_argument('--nmels', type=int, default=41) # nmels = 41개
	parser.add_argument('--nmfcc', type=int, default=40) # nmfcc = 40개
	parser.add_argument('--minmax', type=int, default=1, choices=[0, 1]) # 최대최소 (0,1)사이로 설정
	#parser.add_argument('--model_path', type=str, default='../chkpt/20211005_modelstatedict_labelmodel_epoch_123_ctc.pt')
	#parser.add_argument('--model_path', type=str, default='/home/doyeon/KWS/donut_train/model/20220313_modelstatedict_labelmodel_epoch_193_ctc.pt')
	#parser.add_argument('--model_path', type=str,
	#                     default='/home/hwhan/KeySpot_hw_code/donut_train/chkpt/20220320_modelstatedict_labelmodel_epoch_170_ctc.pt')
	# parser.add_argument('--model_path', type=str,
	#                     default='/home/hwhan/KeySpot_hw_code/donut_train/chkpt/20210330_modelstatedict_labelmodel_epoch_148_ctc.pt')
	parser.add_argument('--model_path', type=str,
						default='/home/dbrms7459/ctc_librispeech/checkpoints/checkpoint_epoch_100')
	return parser


if __name__ == '__main__':
	
	# set dataset parameters
	# arg에 argparser에서 설정한 기본 하이퍼파라미터를 저장한다.
	args = get_parser().parse_args() 
	
	batch_size = args.batchsize
	gpu_number = args.gpu
	feature_type = args.feature
	sampling_rate = args.sr
	frm_length = int(np.floor(0.025 * sampling_rate))
	hop_length = int(np.floor(0.010 * sampling_rate))
	n_fft = args.nfft
	win_type = args.wintype
	n_mels = args.nmels
	n_mfcc = args.nmfcc

	datasetname = args.dataset

	# set gpu
	os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_number)
	device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
	print('device = ', device)

	# set feature type
	if feature_type == 'M':
		feat_ext = torchaudio.transforms.MelSpectrogram(sample_rate=sampling_rate, win_length=frm_length, \
														hop_length=hop_length, n_fft=n_fft, n_mels=n_mels)
	elif feature_type == 'C':
		melargs = dict()
		melargs['win_length'] = frm_length
		melargs['hop_length'] = hop_length
		melargs['n_fft'] = n_fft
		melargs['f_max'] = int(sampling_rate/2)
		melargs['n_mels'] = n_mels
	
		feat_ext = torchaudio.transforms.MFCC(sample_rate=sampling_rate, n_mfcc=n_mfcc, 
												dct_type=2, norm='ortho', melkwargs=melargs)
	feat_ext = feat_ext.to(device)


	# load dataset		
	# libri_rootpath = '/data/LibriSpeech_clean_wav/'
	# libri_rootpath = '/home/hwhan/Database/LibriMix/LibriSpeech/' # 오크 원본
	libri_rootpath = '/home/dbrms7459/audio_data/test-clean/'
	libri_test_filename = ['/home/dbrms7459/ctc_librispeech/data/test_2h.csv']

	test_dataset = LibriSpeechDataset(libri_rootpath, 
                                  	 libri_test_filename, 
                                     feature_type, 
                                     frm_length, hop_length, 
                                     n_fft, sampling_rate, 
                                     win_type, n_mels, 
                                     n_mfcc, device)

	my_collate = my_collate_fortriplet

	test_loader = torch.utils.data.DataLoader(test_dataset, 
                                             batch_size=batch_size, 
                                             shuffle=False,
                                             collate_fn=my_collate, 
                                             num_workers=0, 
                                             pin_memory=True)


	if datasetname == 'libri':
		# 2021.09
		# Original: testset_update_librispeech*
		# V1: testset_update_update_update_librispeech*
		# V2: testset_update_update_librispeech*
		folder_path = '/home/dbrms7459/ctc_librispeech/data/'
		libri_test_filename = sorted(glob(os.path.join(folder_path, 'test_2h.csv')))
		# libri_test_filename = sorted(glob(os.path.join(folder_path, 'testset_update_update_librispeech*')))
		print(libri_test_filename)
		print('=========V2 test===========')
	elif datasetname == 'google_v1':
		enroll_filename = '../data/google/Google_v1_enroll_3_anchors.csv'
		test_filename = '../data/google/Google_v1_enroll_3_test.csv'
	elif datasetname == 'qualcomm':
		test_filename = '../data/qualcomm/qualcomm_testset.csv'
		test_filename = '../data/qualcomm/qualcomm_testset_diff_spk.csv'

	labelmodel = LabelModel()
	labelmodel.to(device)
	labelmodel_path = args.model_path
	labelmodel.load_state_dict(torch.load(labelmodel_path))

	print('Model params. : ', sum([param.nelement() for param in labelmodel.parameters()]))

	loss = torch.nn.CTCLoss()
	if torch.cuda.is_available():
		labelmodel = labelmodel.cuda()
		loss = loss.cuda()

	# evaluation
	#evaluate(test_filename, labelmodel, loss, device)
	
	# if datasetname == 'libri':
	# 	for test_filename in libri_test_filename:
	# 		print('test_filename: ', test_filename)
	# 		evaluate_libriphrase(test_filename, labelmodel, loss, feat_ext, device, minmax=args.minmax)
	# elif datasetname == 'qualcomm':
	# 	evaluate_qualcomm(test_filename, labelmodel, loss, feat_ext, device, minmax=args.minmax)
	# print('------------finish------------')


# 