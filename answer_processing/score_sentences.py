
def get_score(sentence, words, keywords):
    ''' counts how often any words from the question appear in a sentence and returns that as the score '''
    amount = 0
    for word in words:
        if word in sentence:
            if word in keywords: # Woerter gewichten
                amount += (sentence.count(word) * 2)
            else:
                amount += sentence.count(word)
    return amount


def get_score_documents(sentences, words, keywords):
    ''' run through all sentences in all documents and scores them '''
    docu_dict = {}
    for sentence in sentences:
        score = get_score(sentence, words, keywords)
        if score in docu_dict:
            docu_dict[score].append(sentence)
        else:
            docu_dict[score] = [sentence]
    return docu_dict


if __name__ == '__main__':
    pass