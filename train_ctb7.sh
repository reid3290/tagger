nohup python tagger.py train -iter 20 -tb 20 -g 0 -p CTB7 -t train.txt -bt 100  -ng 0 -ws 0 -fn 100 -d dev.txt -wv -cp -gru -m CTB7_5 -emb Embeddings/glove.txt > train_ctb7_5.log &
