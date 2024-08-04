from elasticsearch import Elasticsearch, helpers
import json
from tqdm import tqdm

es = Elasticsearch(['http://localhost:9200'])

# ElasticSearch Einstellungen
settings = {
    "number_of_shards" : 1, #sharding 
    "number_of_replicas" : 2
}

mappings = {
    "properties" : {
        "title" : {"type":"text"}, 
        "text": {
            "type": "text",
            "analyzer": "english", 
            "search_analyzer": "standard" #TODO mal ein anderen analyzer ausprobieren.
        }
    } 
}

es.options(ignore_status=[400,404]).indices.delete(index="wikibase")
es.indices.create(index="wikibase", settings=settings, mappings=mappings)


# Wikibase Datei einlesen und in ElasticSearch packen
liste = []

with open("data/wikibase.jsonl") as file:
    for line in tqdm(file):
        line = json.loads(line)
        line["doc_id"] = int(line["doc_id"])
        liste.append({
            "_index": "wikibase",
            "_id": line["doc_id"],
            "title": line["title"],
            "text": line["text"]
        })
        if len(liste) == 1000:
            helpers.bulk(es, liste)
            #print("1000")
            liste = []

helpers.bulk(es, liste)

es.indices.flush() # persist changes (memory -> disk )
es.indices.refresh() # makes changes available for search
