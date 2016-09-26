#!/bin/bash

#python topicExp.py -s 20news train


#python topicExp.py -p 20news-train-11314-sep281-em100-last.topic.vec 20news train,test
python topicExp.py \
--corpus 20news \
--vocabulary /data/wikipedia/2016-06-21/topic-models/topic.20news.50-1500.sentences/model.dim-200.skip-gram.embedding.restricted.vocab.counts \
--embeddings /data/wikipedia/2016-06-21/topic-models/topic.20news.50-1500.sentences/dim-200.skip-gram.embedding.model.restricted.vocab.embedding \
--set-names train,test \
--max-iterations 100 \
--proportions 20news-train-11314-sep281-em200-best.topic.vec
#python classEval.py 20news topicprop

#python classEval.py 20news topic-wvavg
