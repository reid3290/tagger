nohup python tagger.py train -iter 10 -tb 10 -g 0 -p ud1 -t train.txt -bt 80  -ng 0 -ws 5 -fn 100 -d dev.txt -wv -cp -gru -m ud1-CNN -emb Embeddings/glove.txt > train_ud1_cnn.log &
