nohup python tagger.py train -iter 10 -tb 1 -g 0 -p PFR -t train.txt -bt 80  -ng 0 -ws 5 -fn 100 -d dev.txt -wv -cp -gru -m PFR-CNN -emb Embeddings/glove.txt > train_pfr_cnn.log &
