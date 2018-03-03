nohup python tagger.py train -iter 30 -tb 20 -g 0 -p CTB7 -t train.txt -bt 100  -ng 0 -ws 5 -fn 100 -d dev.txt -wv -cp -gru -m CTB7-CNN -emb Embeddings/glove.txt > train_ctb7_cnn.log &
