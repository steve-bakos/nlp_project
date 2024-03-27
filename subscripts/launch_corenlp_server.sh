PORT=$1

cd tools/stanford-corenlp-full-2016-10-31 && java -Xmx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer \
-serverProperties StanfordCoreNLP-chinese.properties \
-preload tokenize,ssplit,pos,lemma,ner,parse \
-status_port $PORT  -port $PORT -timeout 15000