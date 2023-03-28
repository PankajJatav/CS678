Please run the sentiment_classifier.py with default data
please run the tune.py for all hyper-parameter tuning 

Files and folder:

ReadMe.md  Report  Src

./Report:
Report.pdf

./Src:
best_model.ckpt  models.py         sentiment_classifier.py  tune.py
data             plots             sentiment_data.py
evaluator.py     preprocessing.py  test-blind.output.txt

./Src/data:
dev.txt  test-blind.txt  train.txt  vocab.txt

./Src/plots:
Adagrad  AdamW         emb_dim        GloveT        SGD
Adam     batch_tuning  epochs_tuning  hidden_units

./Src/plots/Adagrad:
loss  result.png

./Src/plots/Adagrad/loss:
0.0001_Adagrad.png  0.01_Adagrad.png  1_Adagrad.png
0.001_Adagrad.png   0.1_Adagrad.png   None_Adagrad.png

./Src/plots/Adam:
loss  result.png

./Src/plots/Adam/loss:
0.0001_Adam.png  0.01_Adam.png  1_Adam.png
0.001_Adam.png   0.1_Adam.png   None_Adam.png

./Src/plots/AdamW:
loss  result.png

./Src/plots/AdamW/loss:
0.0001_AdamW.png  0.01_AdamW.png  1_AdamW.png
0.001_AdamW.png   0.1_AdamW.png   None_AdamW.png

./Src/plots/batch_tuning:
loss  result.png

./Src/plots/batch_tuning/loss:
128_batch.png  16_batch.png  256_batch.png  32_batch.png  64_batch.png

./Src/plots/emb_dim:
loss  result.png

./Src/plots/emb_dim/loss:
1000_emb_dim.png  300_emb_dim.png  50_emb_dim.png
100_emb_dim.png   500_emb_dim.png

./Src/plots/epochs_tuning:
loss  result.png

./Src/plots/epochs_tuning/loss:
10_epochs.png  20_epochs.png  30_epochs.png
15_epochs.png  25_epochs.png  5_epochs.png

./Src/plots/GloveT:
loss  result.png

./Src/plots/GloveT/loss:
False_Glove.png  True_Glove.png

./Src/plots/hidden_units:
loss  result.png

./Src/plots/hidden_units/loss:
1000_hidden_units.png  300_hidden_units.png  50_hidden_units.png
100_hidden_units.png   500_hidden_units.png

./Src/plots/SGD:
loss  result.png

./Src/plots/SGD/loss:
0.0001_SGD.png  0.001_SGD.png  0.01_SGD.png  0.1_SGD.png  1_SGD.png
