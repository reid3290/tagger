nohup python tagger.py train -g 0 -p PFR -t train.txt -bt 80 -d dev.txt -wv -cp -rd -gru -m PFR -emb Embeddings/glove.txt > train_pfr.log &
