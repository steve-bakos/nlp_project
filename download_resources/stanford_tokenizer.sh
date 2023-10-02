set -e

mkdir -p tools

cd tools

if [ ! -d stanford-corenlp-full-2016-10-31 ]; then
    wget http://nlp.stanford.edu/software/stanford-corenlp-full-2016-10-31.zip
    unzip stanford-corenlp-full-2016-10-31.zip
fi

cd stanford-corenlp-full-2016-10-31

if [ ! -f stanford-chinese-corenlp-2016-10-31-models.jar ]; then
    wget http://nlp.stanford.edu/software/stanford-chinese-corenlp-2016-10-31-models.jar
fi

if [ ! -f StanfordCoreNLP-chinese.properties ]; then
    wget https://raw.githubusercontent.com/stanfordnlp/CoreNLP/master/src/edu/stanford/nlp/pipeline/StanfordCoreNLP-chinese.properties 
fi

cd ../..