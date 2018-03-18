nohup python tagger.py train -iter 20 -tb 64 -ed 64 -g 3 -p CTB7 -t train.txt -bt 10  -d dev.txt -cp -gru -m CTB7-5 -emb Embeddings/glove.txt > ctb7-lm-5.log &
