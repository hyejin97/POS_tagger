import string
import math 

##Load train & test files##
train_file = open("WSJ_02-21.pos")
train_list = train_file.readlines()
test_file = open("WSJ_24.words")
test_list = test_file.readlines()

#Morphology
unknown_words = ["UNKNOWN_digit", "UNKNOWN_punct", "UNKNOWN_capital", "UNKNOWN_noun", "UNKNOWN_adj", "UNKNOWN_verb", "UNKNOWN_adv", "UNKNOWN"]
suffix_noun = ["ment", "ness", "age", "ship", "ance", "cy", "ty", "dom", "ee", "ence", "er", "hood", "ion", "action", "ism", "ist", "ity", "ling", "or", "ry", "scape"]
suffix_adj = ["ive", "ful", "ous", "able", "ible", "ly", "ese", "ian", "ic", "ish", "less"]
suffix_verb = ["ize", "ify", "ise", "ate"]
suffix_adv = ["wise", "ward", "wards"]
punctuations = set(string.punctuation)


def tag_unknowns(word):
    '''
    For computing probability of UNKNOWN_WORD
    '''
    if any(word.endswith(suffix) for suffix in suffix_noun):
        return "UNKNOWN_noun"
    elif any(word.endswith(suffix) for suffix in suffix_adj):
        return "UNKNOWN_adj"
    elif any(word.endswith(suffix) for suffix in suffix_verb):
        return "UNKNOWN_verb"
    elif any(word.endswith(suffix) for suffix in suffix_adv):
        return "UNKNOWN_adv"
    elif any(char in punctuations for char in word):
        return "UNKNOWN_punct"
    elif any(char.isdigit() for char in word):
        return "UNKNOWN_digit"
    elif any(char.isupper() for char in word):
        return "UNKNOWN_capital"
    else:
        return "UNKNOWN"
    
def preprocess():    
    '''
    preprocess data
    treat words with freq < 2 as unknown
    tags : list of all possible states(all POS + begin/end of sentences)
    '''
    tags = dict() 
    vocabs = dict() 
    for i in range(len(train_list)):
        split_list = train_list[i].split()
        if len(split_list) == 0:
            continue

        word = split_list[0]
        tag = split_list[1]
        if word not in vocabs:
            vocabs[word] = 1
        else:
            vocabs[word] += 1
        if tag not in tags:
            tags[tag] = 1
        else:
            tags[tag] += 1

    vocab_list = []
    for vocab in vocabs.keys():
        if vocabs[vocab] >= 2:
            vocab_list.append(vocab)
    for unknown in unknown_words:
        vocab_list.append(unknown)
    vocab_list.append("SOS")
    vocab_list.append("EOS")

    tags = list(tags.keys())
    tags.append("Begin_Sent")
    tags.append("End_Sent")
    
    return vocab_list, tags

## train transition and emission probabilities ##
#################################################

def train_HMM(vocab_list, tags, pseudocount=0.001):
    vocab_set = set(vocab_list)
    emission_dict = dict() #Calculating the possible tags for each word
    transition_dict = dict() #Calculating the probabilities of tag bigrams for transition probability  

    for i in range(len(train_list)):
        split_list = train_list[i].split()
        if len(split_list) == 0:
            continue

        word = split_list[0]
        tag = split_list[1]

        if word not in vocab_set: #handling unknown word
            word = tag_unknowns(word)

        ##update emissions
        if tag in emission_dict:
            if word in emission_dict[tag]:
                emission_dict[tag][word] += 1
            else:
                emission_dict[tag][word] = 1
        else:
            emission_dict[tag] = {word : 1}

        ##update transitions
        #if begin of sentence
        if i == 0:
            transition_dict["Begin_Sent"] = {tag : 1}
            emission_dict["Begin_Sent"] = {"SOS" : 1}

        #if end of sentence
        elif i >= len(train_list) - 2:
            if tag in transition_dict:
                transition_dict[tag]["End_Sent"] = 1
            else:
                transition_dict[tag] = {"End_Sent" : 1}
            emission_dict["End_Sent"] = {"EOS" : 1}

        #else
        else:
            if train_list[i+1] == '\n':
                next_split_list = train_list[i+2].split()
                next_tag = next_split_list[1]
            else:
                next_split_list = train_list[i+1].split()
                next_tag = next_split_list[1]

            if tag in transition_dict:
                if next_tag in transition_dict[tag]:
                    transition_dict[tag][next_tag] += 1
                else:
                    transition_dict[tag][next_tag] = 1
            else:
                transition_dict[tag] = {next_tag : 1}

            
    ##convert to probabilities##
    for tag in tags:
        count_tot = 0
        for word in emission_dict[tag]:
            count_tot += emission_dict[tag][word]

        for word in vocab_list:
            count = 0
            if word in emission_dict[tag]:
                count = emission_dict[tag][word]

            emission_dict[tag][word] = (count + pseudocount) / (count_tot + pseudocount * len(vocabs.keys())) #smoothing

    for tag in tags:
        count_tot = 0
        for ftag in tags:  #following tags
            if tag in transition_dict and ftag in transition_dict[tag]:
                count_tot += transition_dict[tag][ftag]

        for ftag in tags:
            count = 0
            if tag in transition_dict:
                if ftag in transition_dict[tag]:
                    count = transition_dict[tag][ftag]
                transition_dict[tag][ftag] = (count + pseudocount) / (count_tot + pseudocount * len(tags)) #smoothing
            else:
                transition_dict[tag] = {ftag : (count + pseudocount) / (count_tot + pseudocount * len(tags))}
    
    return transition_dict, emission_dict
            

def run_test(tags, transition_dict, emission_dict):
    observ = [] #list of words in the test set(sequence of observation)
    observ.append("SOS")
    
    for i in range(len(test_list)):
        if test_list[i] == '\n':
            continue

        test_list[i] = test_list[i].rstrip()

        if test_list[i] not in vocab_set:
            observ.append(tag_unknowns(test_list[i])) #handling unknown
        else:
            observ.append(test_list[i])
            
    observ.append("EOS")

    ##simple 2D transducer##
    #cells represent the likelihood that a particular word is at a particular state
    #emission table for observed words
    emiss_table = []
    for tag in tags:
        cols = []
        for word in observ:
            cols.append(emission_dict[tag][word])
        emiss_table.append(cols)

    max_probs, max_tags = viterbi(observ, emiss_table, transition_dict)
    
    ##Choose the highest POS tags##
    best_idx = [None] * len(observ)
    predicted_tags = [None] * len(observ)

    argmax = max_probs[0][len(observ) - 2]
    best_idx[len(observ) - 2] = 0
    for t in range(1, len(tags)):
        if max_probs[t][len(observ) - 2] > argmax:
            argmax = max_probs[t][len(observ) - 2]
            best_idx[len(observ) - 2] = t

    predicted_tags[len(observ) - 2] = tags[best_idx[len(observ) - 2]]

    for i in range(len(observ) - 2, 1, -1):
        best_idx[i - 1] = max_tags[best_idx[i]][i]
        predicted_tags[i - 1] = tags[best_idx[i - 1]]
        
    return predicted_tags

    
def viterbi(observ, emiss_table, transition_dict):
    '''
    Executing the Viterbi Algorithm
    '''
    max_probs = [[0] * len(observ) for i in range(len(tags))]
    max_tags = [[None] * len(observ) for i in range(len(tags))]

    max_score = 0
    max_i = None

    #intitialize the starting word of sentence
    for row in range(len(tags)):
        s = tags[row]
        if s in transition_dict["Begin_Sent"]:
            score = math.log(emiss_table[row][0]) + math.log(transition_dict["Begin_Sent"][s])
        else:
            score = 0
        max_probs[row][0] = score
        max_tags[row][0] = 0

    for col in range(1, len(observ)):    
        for t in range(len(tags)):  
            max_score = float("-inf")
            max_i = None
            for pt in range(len(tags)):

                score = max_probs[pt][col-1] + math.log(transition_dict[tags[pt]][tags[t]]) + math.log(emiss_table[t][col])

                if score > max_score:
                    max_score = score
                    max_i = pt

            max_probs[t][col] = max_score
            max_tags[t][col] = max_i
        
    return max_probs, max_tags
    
def writefile(predicted_tags):
    '''
    write tagged file
    '''
    predicted_tags = predicted_tags[1:len(predicted_tags)-1] #except sos and eos
    pred_idx = 0

    #add predicted tags for each line    
    for i in range(len(test_list)):
        test_list[i] = test_list[i].rstrip()

        if test_list[i] == '':
            continue
        if pred_idx < len(predicted_tags):
            test_list[i] = test_list[i] + "\t" + predicted_tags[pred_idx]
            pred_idx += 1

    test_list[len(test_list)-1] = "\n"

    output = open(r"result.pos", "w")
    output.write('\n'.join(test_list))
    output.close()

    
if __name__ == '__main__':
    vocab_list, tags = preprocess()
    trans, emis = train_HMM(vocab_list, tags)
    predictions = run_test(tags, trans, emis)
    writefile(predictions)
    