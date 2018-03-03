nohup python tagger.py train -iter 5 -tb 10 -g 0 -p PFR -t train.txt -bt 180  -ng 0 -ws 5 -fn 100 -d dev.txt -wv -cp -gru -m PFR-CNN -emb Embeddings/glove.txt > train_pfr_cnn.log &
