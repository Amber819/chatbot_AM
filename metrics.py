import nltk
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction
cc = SmoothingFunction()

def entity_f1(test_preds, testA):
    scores = []
    for pred, a in zip(test_preds,testA):
        pred_set = set(pred)
        a = a[:a.index(3)]
        true_set = set(a)
        Precision = len(pred_set & true_set)/len(pred_set)
        Recall = len(pred_set & true_set)/len(true_set)
        scores.append(2*Precision*Recall/(Precision + Recall))
    return np.mean(scores)


def per_response(preds, answers):
    acc = 0.0
    for i in len(preds):
        score = 1.0
        pred = preds[i]
        answer = answers[i][:answers[i].index(3)]
        if len(pred) != len(answer):
            score = 0.0
        else:
            for j in len(pred):
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
