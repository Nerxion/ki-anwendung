# Q&A System 
### MIler

### Autoren:  Mike Daudrich und Pablo Schneider

## Ablauf

- wikibase starten
- - run elasicsearch with:
- - - ES_PATH_CONF=my_config ./bin/elasticsearch 
- - - ./bin/elasticsearch
- Wenn elasticsearch noch nicht mit Daten gefüttert ist
- - `wikibase.jsonl` in `./data`Ordner legen
- - `python3 ./wikibase/wiki_base_einlesen.py` ausfuehren
- für fastext `python3 ./data/fasttext/download_fasttext.py` ausfuehren
- - oder fasttext datei in ./data/fasttext Odner ziehen
- File mit qiven quesitons als `qiven_questions.txt` in den `./data` Ordner legen
- Um Q&A laufen zulassen:
- - `python3 main.py` ausfuehren.
- die `result.csv` wird im `./data` Ordner generiert.


