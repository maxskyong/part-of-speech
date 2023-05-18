# Group Members:
# Maximilian Yong Say Kiat
# Alistair Kye-An Tan 
# Lee Jie Long, Bryan 
# Teo Yee Hui 

import re
import pandas as pd

# output_probs.txt generated using mle_probs("twitter_train.txt","output_probs.txt",0.1)
# output_probs2.txt generated using mle_probs2("twitter_train.txt","output_probs2.txt",0.1)
# trans_probs.txt generated using trans_probs("twitter_train.txt","twitter_tags.txt", "trans_probs.txt",0.1)
# trans_probs2.txt generated using trans_probs("twitter_train.txt","twitter_tags.txt", "trans_probs2.txt",0.1)

# Implement the six functions below
def mle_probs(in_data_filename, out_probs_filename, delta):
    
    tags = {}
    tokens = {}
    tokensandtags = {}
    
    with open(in_data_filename, "r", encoding='utf-8') as text:
        for line in text:
            if line == '\n':
                continue
            tokentag = line.split("\t")
            # convert each word to all lowercase since we are currently
            # not case sensitive, for example we would treat "Hello", "hello"
            # and "hElLo" all as the same word.
            token = tokentag[0].lower()
            tag = tokentag[1][:-1]
            if (tag not in tags):
                tags[tag] = 1
                tokensandtags[tag] = {}
            else:
                tags[tag] += 1
            if (token not in tokensandtags[tag]):
                tokensandtags[tag][token] = 1
            else:
                tokensandtags[tag][token] += 1
            if (token not in tokens):
                tokens[token] = 1

    num_words = len(tokens)
    
    with open(out_probs_filename, 'w', encoding='utf-8') as result:
        for tag in tokensandtags:
            for token in tokensandtags[tag]:
                mle_prob = (tokensandtags[tag][token] + delta) / (tags[tag] + delta*(num_words+1))
                line = f'{tag} {token} {mle_prob}' 
                result.write(line)
                result.write('\n')
            unseenprob = delta / (tags[tag] + delta*(num_words+1))
            unseenword = f'{tag} WordDoesNotExist {unseenprob}'
            result.write(unseenword)
            result.write('\n')
    return

# Used to generate output_probs2.txt, no changes to trans_probs2.txt
def mle_probs2(in_data_filename, out_probs_filename, delta):
    
    tags = {}
    tokens = {}
    tokensandtags = {}
    
    with open(in_data_filename, "r", encoding='utf-8') as text:
        for line in text:
            if line == '\n':
                continue
            tokentag = line.split("\t")
            token = tokentag[0].lower()
            # not distinguishing between usernames based on the different trailing IDs after "@user"
            if (token.startswith('@user')):
                token = '@user'
            # not distinguishing between different websites based on the different URLs after "http://"
            if (token.startswith('http://')):
                token = 'http://'
            tag = tokentag[1][:-1]
            if (tag not in tags):
                tags[tag] = 1
                tokensandtags[tag] = {}
            else:
                tags[tag] += 1
            if (token not in tokensandtags[tag]):
                tokensandtags[tag][token] = 1
            else:
                tokensandtags[tag][token] += 1
            if (token not in tokens):
                tokens[token] = 1

    num_words = len(tokens)
    
    with open(out_probs_filename, 'w', encoding='utf-8') as result:
        for tag in tokensandtags:
            for token in tokensandtags[tag]:
                mle_prob = (tokensandtags[tag][token] + delta) / (tags[tag] + delta*(num_words+1))
                line = f'{tag} {token} {mle_prob}' 
                result.write(line)
                result.write('\n')
            unseenprob = delta / (tags[tag] + delta*(num_words+1))
            unseenword = f'{tag} WordDoesNotExist {unseenprob}'
            result.write(unseenword)
            result.write('\n')
    return

def trans_probs(in_data_filename, in_tags_filename, out_probs_filename, delta):
    
    uniquetags = []
    tags = {}
    tagtransitions = {}
    tagtransitions['START'] = {}
    
    with open(in_tags_filename, "r", encoding='utf-8') as text:
        for line in text:
            tag = line.strip()
            uniquetags.append(tag)

    for tag in uniquetags:
        tags[tag] = 0
        tagtransitions[tag] = {}
        tagtransitions['START'][tag] = 0        
        for endtag in uniquetags:
            tagtransitions[tag][endtag] = 0
        tagtransitions[tag]['STOP'] = 0

    with open(in_data_filename, "r", encoding='utf-8') as text:
        position = 0
        prevtag = 'START'
        for line in text:
            if line == '\n':
                tagtransitions[prevtag]['STOP'] += 1
                position = 0
                continue
            tokentag = line.split("\t")
            tag = tokentag[1][:-1]
            tags[tag] += 1
            if (position == 0):
                tagtransitions['START'][tag] += 1
            else:
                tagtransitions[prevtag][tag] += 1
            prevtag = tag
            position += 1

    num_tags = len(uniquetags)
    num_of_examples = sum(tagtransitions['START'].values())

    with open(out_probs_filename, 'w', encoding='utf-8') as result:
        result.write('From To Probability')
        result.write('\n')
        for starttag in tagtransitions:
            for endtag in tagtransitions[starttag]:
                if (starttag == 'START'):
                    initialprob = (tagtransitions[starttag][endtag] + delta) / \
                                  (num_of_examples + delta*(num_tags+1))
                    result.write(f'{starttag} {endtag} {initialprob}')
                    result.write('\n')
                else:
                    transprob = (tagtransitions[starttag][endtag] + delta) / \
                                (tags[starttag] + delta*(num_tags+1))
                    result.write(f'{starttag} {endtag} {transprob}')
                    result.write('\n')
    return


def make_matrix(num_states,seq_length):
    return [[0 for i in range(seq_length)] for j in range(num_states)]


def naive_predict(in_output_probs_filename, in_test_filename, out_prediction_filename):
    probs = pd.read_table(in_output_probs_filename, names=['Tag','Token','MLE_Prob'], delimiter=' ')
    testtags = open(in_test_filename, "r", encoding='utf-8')
    tokens = probs['Token'].values.tolist()
    with open(out_prediction_filename, 'w', encoding='utf-8') as result:
        for line in testtags:
            if line == '\n':
                result.write('\n')
                continue
            token = line.strip()
            token = token.lower()
            if token in tokens:
                filtered = probs[probs['Token'] == token]
                index = filtered['MLE_Prob'].idxmax()
                tag = probs.iloc[index]['Tag']
                result.write(tag)
                result.write('\n')
            else:
                filtered = probs[probs['Token'] == "WordDoesNotExist"]
                index = filtered['MLE_Prob'].idxmax()
                tag = probs.iloc[index]['Tag']
                result.write(tag)
                result.write('\n')
    return

def naive_predict2(in_output_probs_filename, in_train_filename, in_test_filename, out_prediction_filename):
    tags = {}
    
    with open(in_train_filename, "r", encoding='utf-8') as text:
        for line in text:
            if line == '\n':
                continue
            tokentag = line.split("\t")
            tag = tokentag[1][:-1]
            if (tag not in tags):
                tags[tag] = 1
            else:
                tags[tag] += 1

    num_words = sum(tags[tag] for tag in tags)

    probs = pd.read_table(in_output_probs_filename, names=['Tag','Token','MLE_Prob'], delimiter=' ')
    probs['Better_Naive_Prob'] = probs['MLE_Prob']*(probs['Tag'].apply(lambda x: tags[x])/num_words)
    testtags = open(in_test_filename, "r", encoding='utf-8')
    tokens = probs['Token'].values.tolist()
    
    with open(out_prediction_filename, 'w', encoding='utf-8') as result:
        for line in testtags:
            if line == '\n':
                result.write('\n')
                continue
            token = line.strip()
            token = token.lower()
            if token in tokens:
                filtered = probs[probs['Token'] == token]
                index = filtered['Better_Naive_Prob'].idxmax()
                tag = probs.iloc[index]['Tag']
                result.write(tag)
                result.write('\n')
            else:
                filtered = probs[probs['Token'] == "WordDoesNotExist"]
                index = filtered['Better_Naive_Prob'].idxmax()
                tag = probs.iloc[index]['Tag']
                result.write(tag)
                result.write('\n')
    return
    

def viterbi_predict(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                    out_predictions_filename):

    uniquetags = []
    transprobs = {}
    transprobs['START'] = {}
    outputprobs = {}
    
    with open(in_tags_filename, "r", encoding='utf-8') as text:
        for line in text:
            tag = line.strip()
            uniquetags.append(tag)
            transprobs[tag] = {}
            outputprobs[tag] = {}
            
    num_tags = len(uniquetags)
    with open(in_trans_probs_filename, 'r', encoding='utf-8') as text:
        next(text)
        for line in text:
            tagsandprob = line.split(" ")
            fromtag = tagsandprob[0]
            totag = tagsandprob[1]
            prob = tagsandprob[2][:-1]
            transprobs[fromtag][totag] = prob

    with open(in_output_probs_filename, 'r', encoding='utf-8') as text:
        for line in text:
            tagtokenprob = line.split(" ")
            tag = tagtokenprob[0]
            token = tagtokenprob[1]
            prob = tagtokenprob[2][:-1]
            outputprobs[tag][token] = prob

    testtokens = open(in_test_filename, "r", encoding='utf-8')
    
    with open(out_predictions_filename, 'w', encoding='utf-8') as result:
        seqlength = 0
        currtokens = []
        for line in testtokens:
            if line == '\n':
                dpmatrix = make_matrix(num_tags,seqlength)
                bpmatrix = make_matrix(num_tags,seqlength)
                starttoken = currtokens[0]
                
                for i in range(num_tags):
                    currtransprob = float(transprobs['START'][uniquetags[i]])
                    if starttoken in outputprobs[uniquetags[i]]:
                        curroutprob = float(outputprobs[uniquetags[i]][starttoken])
                    else:
                        curroutprob = float(outputprobs[uniquetags[i]]['WordDoesNotExist'])                        
                    dpmatrix[i][0] = currtransprob*curroutprob
                    bpmatrix[i][0] = 'START'

                for i in range(1, len(currtokens)):
                    for j in range(num_tags):
                        maxprob = 0
                        maxarg = None
                        for k in range(num_tags):
                            if currtokens[i] in outputprobs[uniquetags[j]]:
                                curroutprob = float(outputprobs[uniquetags[j]][currtokens[i]])
                            else:
                                curroutprob = float(outputprobs[uniquetags[j]]['WordDoesNotExist'])
                            currtransprob = float(transprobs[uniquetags[k]][uniquetags[j]])
                            dpprob = dpmatrix[k][i-1]*currtransprob*curroutprob
                            if (dpprob > maxprob):
                                maxprob = dpprob
                                maxarg = k
                        dpmatrix[j][i] = maxprob
                        bpmatrix[j][i] = maxarg

                finalmaxprob = 0
                finalbp = None
                for i in range(num_tags):
                    finaldpprob = dpmatrix[i][seqlength - 1]
                    finaltransprob = float(transprobs[uniquetags[i]]['STOP'])
                    finalprob = finaldpprob*finaltransprob
                    if (finalprob > finalmaxprob):
                        finalmaxprob =  finalprob
                        finalbp = i

                path = []
                path.append(finalbp)
                currbp = finalbp
                backtrack_counter = seqlength - 1
                while (backtrack_counter > 0):
                    currbp = bpmatrix[currbp][backtrack_counter]
                    path.append(currbp)
                    backtrack_counter -= 1
                path.reverse()
                
                for state in path:
                    result.write(uniquetags[state])
                    result.write('\n')
                currtokens = []
                seqlength = 0
                result.write('\n')
            else:
                token = line.strip()
                token = token.lower()
                currtokens.append(token)
                seqlength += 1
    return

def viterbi_predict2(in_tags_filename, in_trans_probs_filename, in_output_probs_filename, in_test_filename,
                     out_predictions_filename):
    
    uniquetags = []
    transprobs = {}
    transprobs['START'] = {}
    outputprobs = {}
    
    with open(in_tags_filename, "r", encoding='utf-8') as text:
        for line in text:
            tag = line.strip()
            uniquetags.append(tag)
            transprobs[tag] = {}
            outputprobs[tag] = {}
            
    num_tags = len(uniquetags)
    with open(in_trans_probs_filename, 'r', encoding='utf-8') as text:
        next(text)
        for line in text:
            tagsandprob = line.split(" ")
            fromtag = tagsandprob[0]
            totag = tagsandprob[1]
            prob = tagsandprob[2][:-1]
            transprobs[fromtag][totag] = prob

    with open(in_output_probs_filename, 'r', encoding='utf-8') as text:
        for line in text:
            tagtokenprob = line.split(" ")
            tag = tagtokenprob[0]
            token = tagtokenprob[1]
            prob = tagtokenprob[2][:-1]
            outputprobs[tag][token] = prob

    testtokens = open(in_test_filename, "r", encoding='utf-8')
    
    with open(out_predictions_filename, 'w', encoding='utf-8') as result:
        seqlength = 0
        currtokens = []
        for line in testtokens:
            if line == '\n':
                dpmatrix = make_matrix(num_tags,seqlength)
                bpmatrix = make_matrix(num_tags,seqlength)
                starttoken = currtokens[0]
                
                for i in range(num_tags):
                    currtransprob = float(transprobs['START'][uniquetags[i]])
                    if starttoken in outputprobs[uniquetags[i]]:
                        curroutprob = float(outputprobs[uniquetags[i]][starttoken])
                    else:
                        curroutprob = float(outputprobs[uniquetags[i]]['WordDoesNotExist'])                        
                    dpmatrix[i][0] = currtransprob*curroutprob
                    bpmatrix[i][0] = 'START'

                for i in range(1, len(currtokens)):
                    for j in range(num_tags):
                        maxprob = 0
                        maxarg = None
                        for k in range(num_tags):
                            if currtokens[i] in outputprobs[uniquetags[j]]:
                                curroutprob = float(outputprobs[uniquetags[j]][currtokens[i]])
                            else:
                                curroutprob = float(outputprobs[uniquetags[j]]['WordDoesNotExist'])
                            currtransprob = float(transprobs[uniquetags[k]][uniquetags[j]])
                            dpprob = dpmatrix[k][i-1]*currtransprob*curroutprob
                            if (dpprob > maxprob):
                                maxprob = dpprob
                                maxarg = k
                        dpmatrix[j][i] = maxprob
                        bpmatrix[j][i] = maxarg

                finalmaxprob = 0
                finalbp = None
                for i in range(num_tags):
                    finaldpprob = dpmatrix[i][seqlength - 1]
                    finaltransprob = float(transprobs[uniquetags[i]]['STOP'])
                    finalprob = finaldpprob*finaltransprob
                    if (finalprob > finalmaxprob):
                        finalmaxprob =  finalprob
                        finalbp = i

                path = []
                path.append(finalbp)
                currbp = finalbp
                backtrack_counter = seqlength - 1
                while (backtrack_counter > 0):
                    currbp = bpmatrix[currbp][backtrack_counter]
                    path.append(currbp)
                    backtrack_counter -= 1
                path.reverse()
                
                for state in path:
                    result.write(uniquetags[state])
                    result.write('\n')
                currtokens = []
                seqlength = 0
                result.write('\n')
            else:
                token = line.strip()
                token = token.lower()
                # not distinguishing between usernames based on the different trailing IDs after "@user"
                if (token.startswith('@user')):
                    token = '@user'
                # not distinguishing between different websites based on the different URLs after "http://"
                if (token.startswith('http://')):
                    token = 'http://'
                currtokens.append(token)
                seqlength += 1
    return


def evaluate(in_prediction_filename, in_answer_filename):
    """Do not change this method"""
    with open(in_prediction_filename) as fin:
        predicted_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    with open(in_answer_filename) as fin:
        ground_truth_tags = [l.strip() for l in fin.readlines() if len(l.strip()) != 0]

    assert len(predicted_tags) == len(ground_truth_tags)
    correct = 0
    for pred, truth in zip(predicted_tags, ground_truth_tags):
        if pred == truth: correct += 1
    return correct, len(predicted_tags), correct/len(predicted_tags)



def run():
    '''
    You should not have to change the code in this method. We will use it to execute and evaluate your code.
    You can of course comment out the parts that are not relevant to the task that you are working on, but make sure to
    uncomment them later.
    This sequence of code corresponds to the sequence of questions in your project handout.
    '''

    ddir = './' #your working dir

    in_train_filename = f'{ddir}/twitter_train.txt'

    naive_output_probs_filename = f'{ddir}/naive_output_probs.txt'

    in_test_filename = f'{ddir}/twitter_dev_no_tag.txt'
    in_ans_filename  = f'{ddir}/twitter_dev_ans.txt'
    naive_prediction_filename = f'{ddir}/naive_predictions.txt'
    naive_predict(naive_output_probs_filename, in_test_filename, naive_prediction_filename)
    correct, total, acc = evaluate(naive_prediction_filename, in_ans_filename)
    print(f'Naive prediction accuracy:     {correct}/{total} = {acc}')

    naive_prediction_filename2 = f'{ddir}/naive_predictions2.txt'
    naive_predict2(naive_output_probs_filename, in_train_filename, in_test_filename, naive_prediction_filename2)
    correct, total, acc = evaluate(naive_prediction_filename2, in_ans_filename)
    print(f'Naive prediction2 accuracy:    {correct}/{total} = {acc}')

    trans_probs_filename =  f'{ddir}/trans_probs.txt'
    output_probs_filename = f'{ddir}/output_probs.txt'

    in_tags_filename = f'{ddir}/twitter_tags.txt'
    viterbi_predictions_filename = f'{ddir}/viterbi_predictions.txt'
    viterbi_predict(in_tags_filename, trans_probs_filename, output_probs_filename, in_test_filename,
                    viterbi_predictions_filename)
    correct, total, acc = evaluate(viterbi_predictions_filename, in_ans_filename)
    print(f'Viterbi prediction accuracy:   {correct}/{total} = {acc}')

    trans_probs_filename2 =  f'{ddir}/trans_probs2.txt'
    output_probs_filename2 = f'{ddir}/output_probs2.txt'

    viterbi_predictions_filename2 = f'{ddir}/viterbi_predictions2.txt'
    viterbi_predict2(in_tags_filename, trans_probs_filename2, output_probs_filename2, in_test_filename,
                     viterbi_predictions_filename2)
    correct, total, acc = evaluate(viterbi_predictions_filename2, in_ans_filename)
    print(f'Viterbi2 prediction accuracy:  {correct}/{total} = {acc}')
    


if __name__ == '__main__':
    run()
