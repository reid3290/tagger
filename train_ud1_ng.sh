nohup python tagger.py train -iter 10 -tb 10 -g 0 -p ud1 -t train.txt -bt 80  -ng 3 -ws 0 -fn 0 -d dev.txt -wv -cp -gru -m ud1-NG -emb Embeddings/glove.txt > train_ud1_ng.log &
