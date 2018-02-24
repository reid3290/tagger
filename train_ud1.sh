nohup python tagger.py train -g 0 -p ud1 -t train.txt -bt 80 -d dev.txt -ws 0 -wv -cp -rd -gru -m ud1 -emb Embeddings/glove.txt > train_ud1.log &
