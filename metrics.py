import nltk
import numpy as np
import os
from nltk.translate.bleu_score import SmoothingFunction
#from seq2seq_dialog import *
cc = SmoothingFunction()

def get_entities():
    kb_data_dir="data/kb-task6-dstc2/"
    files=os.listdir(kb_data_dir)
    files = [os.path.join(kb_data_dir, f) for f in files]
    entities=[]
    for f in files:
        with open(f) as f:
            for line in f.readlines():
                l=line.rstrip().split('\t')
                entities+=l
    return entities

def entity_f1(id2word,test_preds, testA):
    entities=get_entities()
    scores = []
    for pred, a in zip(test_preds,testA):
        pred_set = set(pred)
        a = a[:a.index(3)]
        true_set = set(a)
        if len(pred_set) is 0:
            Precision = 0
            Recall = 0
        else:
            pred_enttys=[]
            true_enttys=[]
            for one_id in pred_set:
                one = id2word[one_id]
                if one in entities:
                    pred_enttys.append(one)
            for ano_id in true_set:
                ano=id2word[ano_id]
                if ano in entities:
                    true_enttys.append(ano)
            pred_enttyset=set(pred_enttys)
            true_enttyset=set(true_enttys)
            if len(true_enttyset) is 0:
                continue
            else:
                if len(pred_enttyset) is 0:
                    Precision = 0
                    Recall = 0
                else:
                    right=pred_enttyset&true_enttyset
                    Precision = len(right)/len(pred_enttyset)
                    Recall = len(right)/len(true_enttyset)
        if Precision + Recall == 0:
            scores.append(0.0)
        else:
            scores.append(2*Precision*Recall/(Precision + Recall))
        print ('len of scores:',len(scores))
    return np.mean(scores)


def per_response(preds, answers):
    acc = 0.0
    for i in range(len(preds)):
        score = 1.0
        pred = preds[i]
        answer = answers[i][:answers[i].index(3)]
        if len(pred) != len(answer):
            score = 0.0
        else:
            for j in range(len(pred)):
                if pred[j]!=answer[j]:
                    score = 0.0
                    break
        acc += score
    avg_score = acc/len(preds)
    return avg_score


def bleu_score(preds, answers):
    bleu = []
    for pred, answer in zip(preds, answers):
        answer = answer[:answer.index(3)]
        bleu.append(nltk.translate.bleu_score.sentence_bleu([pred], answer))
    return np.mean(bleu)
