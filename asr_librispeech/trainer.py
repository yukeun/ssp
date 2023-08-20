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
from torch.utils.tensorboard import SummaryWriter
import pdb
# custom package
#from utils.evaluation_metrics import confusion_matrix, det_curve, plot_det_curve
import matplotlib
matplotlib.use('Agg')

def fit(train_loader, val_loader, model, loss_ctc, optimizer, scheduler, \
        batch_size, n_epochs, device, total_loss=None, last_epoch=None):

    history_lst = []
    # 결과를 저장하는 리스트
    
    now = datetime.datetime.now() 
    # 현재 시각을 얻고 싶다면 datetime 클래스의 now() 메서드를 사용하면 됩니다. 
    Today = now.strftime('%Y%m%d')
    # now = datetime.datetime.now()
    # print(now)          # 2015-04-19 12:11:32.669083
 
    # nowDate = now.strftime('%Y-%m-%d')
    # print(nowDate)      # 2015-04-19

    if last_epoch is not None:
        # epoch가 이전에 존재했다면 이전 에포크부터 시작
        start_epoch = last_epoch
    else:
        start_epoch = 0

    if total_loss is not None:
        min_loss = total_loss
    else:
        min_loss = 1000

    # Open tensorboard
    writer = SummaryWriter(log_dir="logs")

    iteration = 0
    for epoch in range(start_epoch, n_epochs):

        dic_history = {'epoch': '', 'train_total_loss': '', 
                       'train_per': '', 'val_total_loss': '', 
                       'val_per': ''}    
    
        model.train()
        # epoch만큼 모델을 train시키기
        total_loss = 0
        # 전체 loss는 처음엔 0이라고 가정
        
        print('===========Start training============')
        print(len(train_loader))
        for batch_idx, (data, data_length) in tqdm(enumerate(train_loader)):
            # batch_idx : batch index
            # train_loader에서 출력하는 data와 data_length는 뭘까
            # data = {'acoustic': feat, 'text': torch.LongTensor(label)}
			# acoustic flac file에서 뽑은 특징들을 text에 매핑한다.
            # data_length : len(self.df_word)
		    # dataset의 파일의 갯수를 출력
            if len(data[0]) != batch_size: 
                # 우리는 default hyperparameter로 batchsize를 32로 지정함
                # batch_size는 
                print('len(data[0]) =', len(data[0])) 
                print('here-----------')
                continue

            (acoustic_x, ctc_label) = data[0], data[1]
            # data[0]은 data dictionary에서 음성파일 feature를 뜻하고, data[1]은 
            (acoustic_length, text_length) = data_length[0], data_length[1]
            acoustic_x = torch.FloatTensor(acoustic_x)
            acoustic_length = torch.LongTensor(acoustic_length)
            text_length = torch.LongTensor(text_length)
            if torch.cuda.is_available():
                acoustic_x = acoustic_x.to(device)
                ctc_label = ctc_label.to(device)

            posterior_prob = model(acoustic_x, acoustic_length)
            # 확률 값을 구하는 듯.

            optimizer.zero_grad()

            ctc_loss = loss_ctc(posterior_prob, ctc_label, 
                                acoustic_length, text_length)
            # ctc의 경우 softmax output, ctclabel, 그리고 음성x의 길이 및 text_legnth를 입력
            # loss를 구한다음 backpropagation을 진행한다.
            # ctc_loss를 바탕으로 backpropagation 진행

            ctc_loss.backward()
            optimizer.step()
            # update weight
            
            total_loss += ctc_loss.item()
            # .item() : 1개의 값을 가진 텐서의 값을 가져오는 함수
            writer.add_scalar('train loss per iteration', ctc_loss.item(), iteration)
            iteration += 1

        total_loss /= ((batch_idx + 1) * 2)
        message = 'epoch: {}/{} : Training loss = {:.4f}'.format(epoch + 1, 
                                                                 n_epochs, 
                                                                 total_loss)        
        print(message)
        writer.add_scalar('train loss', total_loss, epoch)
        # validation stage
        val_loss, val_per = evaluate(val_loader, model, loss_ctc, 
                                     batch_size, epoch + 1, device)
        message = 'epoch: {}/{} : Val loss = {:.4f}, Val per = {:.2f}%'.format(epoch + 1, n_epochs, val_loss, val_per * 100.)
        print(message)
        writer.add_scalar('valid loss', val_loss, epoch)

        # Save loss and accuracy.
        dic_history['epoch'] = epoch
        dic_history['train_total_loss'] = total_loss
        dic_history['val_total_loss'] = val_loss
        dic_history['val_per'] = val_per
        scheduler.step(val_loss)
        for param_group in optimizer.param_groups:
            print('lr: ', param_group['lr'])

        history_lst.append(dic_history)

        # Save stage
        if epoch == 0:
            min_loss = val_loss

        # Save model
     
        torch.save({'epoch': epoch,
                    'model_state_dict': model.state_dict(), 
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': val_loss},
                    './checkpoints/' + 'checkpoint_labelmodel_' + str(epoch))
        min_loss = val_loss

        # save performance
        df_history = pd.DataFrame(history_lst)
        #df_history.to_csv('./result/' + str(Today) + '_' + str(networkname) + '_' + str(lossname) + '_result.csv', index=None)
        df_history.to_csv('./result/' + str(Today) + '_labelmodel_ctc_result.csv', index=None)

    # save total loss learning curve
    fig, loss_ax = plt.subplots()
    loss_ax.plot(df_history['train_total_loss'], color='#5CD1E5', label='train loss')
    loss_ax.plot(df_history['val_total_loss'], color='r', label = 'val loss')
    loss_ax.set_title('Loss')
    loss_ax.set_xlabel('epoch')
    #loss_ax.set_ylabel('loss')
    loss_ax.set_xlim([0, epoch - 1])
    loss_ax.set_ylim([0, max(max(df_history['train_total_loss']), max(df_history['val_total_loss']))])
    lgd = loss_ax.legend(bbox_to_anchor=(1.01, 1.02))
    #plt.savefig('./result/' + str(Today) + '_' + str(networkname) + '_loss_curve.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.savefig('./result/' + str(Today) + '_labelmodel_loss_curve.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


    # save acc curve
    fig, acc_ax = plt.subplots()

    acc_ax.plot(df_history['val_per'], 'r', label='val accuracy')
    acc_ax.set_title('Error rate')
    acc_ax.set_xlabel('epoch')
    acc_ax.set_xlim([0, epoch - 1])
    acc_ax.set_ylim([0, 1.1])
    lgd = acc_ax.legend(bbox_to_anchor=(1.01, 1.02))
    plt.savefig('./result/' + str(Today) + '_labelmodel_per_curve.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf() 
        
    writer.close()   



def evaluate(test_loader, model, loss_ctc, batch_size, epoch, device):
    model.eval()

    eval_pred = 0
    eval_loss = 0

    label_dict = generate_int_label_dict()
    # loss specifically
    emb_loss_sum = 0
    cls_loss_sum = 0

    scores = []
    labels = []

    y_pred = []
    y_score = []
    y_target = []

    print('===========Start evaluating============')
    with torch.no_grad():
        for batch_idx, (data, data_length) in tqdm(enumerate(test_loader)):
            if batch_size != len(data[0]):
                print('len(data[0]) = ', len(data[0]))
                continue


            #posterior_prob = model(data, data_length)
            (acoustic_x, ctc_label) = data[0], data[1]
            (acoustic_length, text_length) = data_length[0], data_length[1]
            #print(data_length)
            acoustic_x = torch.FloatTensor(acoustic_x)
            acoustic_length = torch.LongTensor(acoustic_length)
            text_length = torch.LongTensor(text_length)
            if torch.cuda.is_available():
                acoustic_x = acoustic_x.to(device)
                ctc_label = ctc_label.to(device)
            #    acoustic_x = acoustic_x.cuda()
            #    ctc_label = ctc_label.cuda()
 
            #start_time = time.time()
            posterior_prob = model(acoustic_x, acoustic_length)
            ctc_label = data[1]

            ctc_loss = loss_ctc(posterior_prob, ctc_label, acoustic_length, text_length)
            eval_loss += ctc_loss.item()
            decoded_predict, decoded_target = GreedyDecoder(posterior_prob, ctc_label, text_length, label_dict)
            #print("Decoded (predict)", decoded_predict)
            #print("Decoded (target)", decoded_target)
            for j in range(len(decoded_predict)):
                #eval_pred += per(decoded_target[j], decoded_predict[j]) / ctc_label.size(0)
                eval_pred += per(decoded_target[j], decoded_predict[j]) / ctc_label.size(0)
            #eval_pred += p_pred.eq(y_p_cls).sum().item() / p_pred.size()[0]

        eval_loss /= (len(test_loader))
        eval_pred /= (len(test_loader))

        print('eval_per: ', eval_pred * 100)

    return eval_loss, eval_pred


def generate_int_label_dict():
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
    return label_dict


def int_to_text(labels, label_dict):
        decoded = []
        for i in range(len(labels)):
                decoded.append(label_dict[labels[i]])
        return decoded



def GreedyDecoder(output, labels, label_lengths, label_dict, blank_label=0, collapse_repeated=True):

    arg_maxes = torch.argmax(output.permute(1, 0, 2), dim=2)
    
    decodes = []
    targets = []
    for i, args in enumerate(arg_maxes):
        decode = []
        targets.append(int_to_text(labels[i][:label_lengths[i]].tolist(), label_dict))
        for j, index in enumerate(args):
            if index != blank_label:
                if collapse_repeated and j != 0 and index == args[j -1]:
                    continue
                decode.append(index.item())
        decodes.append(int_to_text(decode, label_dict))
    return decodes, targets


def save_model(model, filename):
    with open(filename, 'wb') as f:
        torch.save(model, f)
        print('%s saved.'%filename)


#==========================Evaluation metrics============================
def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def cer(reference, hypothesis, ignore_case=False, remove_space=False):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    edit_distance, ref_len = char_errors(reference, hypothesis, ignore_case,
                                         remove_space)

    if ref_len == 0:
        raise ValueError("Length of reference should be greater than 0.")

    cer = float(edit_distance) / ref_len
    return cer

def avg_wer(wer_scores, combined_ref_len):
    return float(sum(wer_scores)) / float(combined_ref_len)


def _levenshtein_distance(ref, hyp):
    """Levenshtein distance is a string metric for measuring the difference
    between two sequences. Informally, the levenshtein disctance is defined as
    the minimum number of single-character edits (substitutions, insertions or
    deletions) required to change one word into the other. We can naturally
    extend the edits to word level when calculate levenshtein disctance for
    two sentences.
    """
    m = len(ref)
    n = len(hyp)

    # special case
    if ref == hyp:
        return 0
    if m == 0:
        return n
    if n == 0:
        return m

    if m < n:
        ref, hyp = hyp, ref
        m, n = n, m

    # use O(min(m, n)) space
    distance = np.zeros((2, n + 1), dtype=np.int32)

    # initialize distance matrix
    for j in range(0,n + 1):
        distance[0][j] = j

    # calculate levenshtein distance
    for i in range(1, m + 1):
        prev_row_idx = (i - 1) % 2
        cur_row_idx = i % 2
        distance[cur_row_idx][0] = i
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                distance[cur_row_idx][j] = distance[prev_row_idx][j - 1]
            else:
                s_num = distance[prev_row_idx][j - 1] + 1
                i_num = distance[cur_row_idx][j - 1] + 1
                d_num = distance[prev_row_idx][j] + 1
                distance[cur_row_idx][j] = min(s_num, i_num, d_num)

    return distance[m % 2][n]


def word_errors(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in word-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Levenshtein distance and word number of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    ref_words = reference.split(delimiter)
    hyp_words = hypothesis.split(delimiter)

    edit_distance = _levenshtein_distance(ref_words, hyp_words)
    return float(edit_distance), len(ref_words)


def char_errors(reference, hypothesis, ignore_case=False, remove_space=False):
    """Compute the levenshtein distance between reference sequence and
    hypothesis sequence in char-level.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Levenshtein distance and length of reference sentence.
    :rtype: list
    """
    if ignore_case == True:
        reference = reference.lower()
        hypothesis = hypothesis.lower()

    join_char = ' '
    if remove_space == True:
        join_char = ''

    reference = join_char.join(filter(None, reference.split(' ')))
    hypothesis = join_char.join(filter(None, hypothesis.split(' ')))

    edit_distance = _levenshtein_distance(reference, hypothesis)
    return float(edit_distance), len(reference)


def wer(reference, hypothesis, ignore_case=False, delimiter=' '):
    """Calculate word error rate (WER). WER compares reference text and
    hypothesis text in word-level. WER is defined as:
    .. math::
        WER = (Sw + Dw + Iw) / Nw
    where
    .. code-block:: text
        Sw is the number of words subsituted,
        Dw is the number of words deleted,
        Iw is the number of words inserted,
        Nw is the number of words in the reference
    We can use levenshtein distance to calculate WER. Please draw an attention
    that empty items will be removed when splitting sentences by delimiter.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param delimiter: Delimiter of input sentences.
    :type delimiter: char
    :return: Word error rate.
    :rtype: float
    :raises ValueError: If word number of reference is zero.
    """
    edit_distance, ref_len = word_errors(reference, hypothesis, ignore_case,
                                         delimiter)

    if ref_len == 0:
        raise ValueError("Reference's word number should be greater than 0.")

    wer = float(edit_distance) / ref_len
    return wer


def per(reference, hypothesis):
    """Calculate charactor error rate (CER). CER compares reference text and
    hypothesis text in char-level. CER is defined as:
    .. math::
        CER = (Sc + Dc + Ic) / Nc
    where
    .. code-block:: text
        Sc is the number of characters substituted,
        Dc is the number of characters deleted,
        Ic is the number of characters inserted
        Nc is the number of characters in the reference
    We can use levenshtein distance to calculate CER. Chinese input should be
    encoded to unicode. Please draw an attention that the leading and tailing
    space characters will be truncated and multiple consecutive space
    characters in a sentence will be replaced by one space character.
    :param reference: The reference sentence.
    :type reference: basestring
    :param hypothesis: The hypothesis sentence.
    :type hypothesis: basestring
    :param ignore_case: Whether case-sensitive or not.
    :type ignore_case: bool
    :param remove_space: Whether remove internal space characters
    :type remove_space: bool
    :return: Character error rate.
    :rtype: float
    :raises ValueError: If the reference length is zero.
    """
    #ref_phones = reference.split(delimiter)
    #hyp_phones = hypothesis.split(delimiter)
    ref_phones, hyp_phones = reference, hypothesis
    edit_distance = _levenshtein_distance(ref_phones, hyp_phones)

    per = float(edit_distance) / len(ref_phones)
    return per
