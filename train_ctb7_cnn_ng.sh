nohup python tagger.py train -iter 10 -tb 5 -g 0 -p CTB7 -t train.txt -bt 200  -ng 3 -ws 5 -fn 100 -d dev.txt -wv -cp -gru -m CTB7-CNN-NG -emb Embeddings/glove.txt > train_ctb7_cnn_ng.log &
