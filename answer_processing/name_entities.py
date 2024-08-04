from answer_processing.score_sentences import get_score_documents
# from similar_sentences import get_score_documents

def get_named_entities_from_sentences(sentences):
    ''' returns named entities of a list of sentences '''
    entity_dicc = dict()

    for sentence in sentences:
        for word in sentence:
            if word.get_ne() not in entity_dicc.keys():
                entity_dicc[word.get_ne()] = [word.word]
            else:
                entity_dicc[word.get_ne()].append(word.word)

    return entity_dicc


def get_named_entities(documents, words, keywords):
    ''' scores all sentences, then sorts them by score and returns only the highest scoring sentences '''
    scores = get_score_documents(documents, words, keywords)
    scores = sorted(scores.items(), reverse=True)
    return get_named_entities_from_sentences(scores[0][1])