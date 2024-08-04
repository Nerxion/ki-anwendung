

from passage_retrival.read_data_wikibase import wikibase_manager
from passage_retrival.filter_document import clean_documents, documents_to_sentences, filter_sentences_similarity, stem_sentences

def file_to_list(path):
    ''' reads results from file into an array '''
    documents = []
    with open(path, 'r') as file:
        for line in file:
            documents.append(line)
    return documents


def get_wiki_result(question, size):
    ''' searches the wikibase for all documents fitting to the given question '''
    wiki = wikibase_manager(question, size)
    wiki_result = wiki.results_to_list()
    
    return wiki_result


def passage_retrieval(question, min_similarity=0.5, size=100):
    ''' main function of the passage retrieval, returns the found documents formatted as sentences '''
    documents = get_wiki_result(question, size)
    documents = clean_documents(documents)
    documents_sentences = documents_to_sentences(documents)
    documents_sentences = filter_sentences_similarity(documents_sentences, question, min_similarity)
    documents_sentences = stem_sentences(documents_sentences)
    return documents_sentences
    

