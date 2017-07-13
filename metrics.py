import nltk
def per_response(preds, answers):
    score=0
    for i in len(preds):
        pred=preds[i]
        answer=answers[i]
        if len(pred)!=len(answer):
            score=0
        else:
            for j in len(pred):
                if pred[i]==answer[i]:
                    score+=1
                else:
                    score=0
    batch_score=score/len(preds)
    return batch_score

def bleu_score(preds, answers):
    bleu_score=0
    list_preds=[preds]
    list_answers=[answers]
    bleu_score+=nltk.translate.bleu_score.corpus_bleu(list_answers,list_preds)
    return bleu_score



