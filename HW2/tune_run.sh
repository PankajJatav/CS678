python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --epochs 5 --dev_out sst-dev-ft-output-5e.txt --test_out sst-test-ft-output-5e.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --epochs 10 --dev_out sst-dev-ft-output-10e.txt --test_out sst-test-ft-output-10e.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --epochs 15 --dev_out sst-dev-ft-output-15e.txt --test_out sst-test-ft-output-15e.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --epochs 20 --dev_out sst-dev-ft-output-20e.txt --test_out sst-test-ft-output-20e.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --epochs 25 --dev_out sst-dev-ft-output-25e.txt --test_out sst-test-ft-output-25e.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --hidden_dropout_prob=0.1 --dev_out sst-dev-ft-output-d1.txt --test_out sst-test-ft-output-d1.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --hidden_dropout_prob=0.3 --dev_out sst-dev-ft-output-d3.txt --test_out sst-test-ft-output-d3.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --hidden_dropout_prob=0.5 --dev_out sst-dev-ft-output-d5.txt --test_out sst-test-ft-output-d5.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --hidden_dropout_prob=0.7 --dev_out sst-dev-ft-output-d7.txt --test_out sst-test-ft-output-d7.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --hidden_dropout_prob=0.9 --dev_out sst-dev-ft-output-d9.txt --test_out sst-test-ft-output-d9.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --lr 0.5 --dev_out sst-dev-ft-output-lr1.txt --test_out sst-test-ft-output-lr1.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --lr 1e-1 --dev_out sst-dev-ft-output-lr2.txt --test_out sst-test-ft-output-lr2.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --lr 1e-3 --dev_out sst-dev-ft-output-lr3.txt --test_out sst-test-ft-output-lr3.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --lr 1e-5 --dev_out sst-dev-ft-output-lr4.txt --test_out sst-test-ft-output-lr4.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --lr 1e-7 --dev_out sst-dev-ft-output-lr5.txt --test_out sst-test-ft-output-lr5.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --batch_size 2  --dev_out sst-dev-ft-output-b2.txt --test_out sst-test-ft-output-b2.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --batch_size 4  --dev_out sst-dev-ft-output-b4.txt --test_out sst-test-ft-output-b4.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --batch_size 8  --dev_out sst-dev-ft-output-b8.txt --test_out sst-test-ft-output-b8.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --batch_size 16  --dev_out sst-dev-ft-output-b16.txt --test_out sst-test-ft-output-b16.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --batch_size 32  --dev_out sst-dev-ft-output-b32.txt --test_out sst-test-ft-output-b32.txt

python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --b1 0.9 --b2 0.98  --eps 10e-9 --dev_out sst-dev-ft-output-b98_e9.txt --test_out sst-test-ft-output-b98_e9.txt

python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --use_paper_adam --warm_up 2000 --b1 0.9 --b2 0.98 --eps 10e-9 --dev_out sst-dev-ft-output-w2.txt --test_out sst-test-ft-output-w2.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --use_paper_adam --warm_up 4000 --b1 0.9 --b2 0.98 --eps 10e-9 --dev_out sst-dev-ft-output-w4.txt --test_out sst-test-ft-output-w4.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --use_paper_adam --warm_up 8000 --b1 0.9 --b2 0.98 --eps 10e-9 --dev_out sst-dev-ft-output-w8.txt --test_out sst-test-ft-output-w8.txt

python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --hidden_dropout_prob=0.7 --b1 0.9 --b2 0.98  --eps 10e-9 --dev_out sst-dev-ft-output-b98_e9d.txt --test_out sst-test-ft-output-b98_e9d.txt

python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --option finetune --b1 0.9 --b2 0.98  --eps 10e-9 --dev_out sst-dev-output.txt --test_out sst-test-output.txt
python classifier.py  --use_gpu --dev data/sst-dev.txt --train data/sst-train.txt --test data/sst-test.txt --b1 0.9 --b2 0.98  --eps 10e-9 --dev_out sst-pt-dev-output.txt --test_out sst-pt-test-output.txt

python classifier.py  --use_gpu --option finetune --b1 0.9 --b2 0.98  --eps 10e-9 --dev_out  cfimdb-dev-output.txt --test_out cfimdb-test-output.txt
python classifier.py  --use_gpu --b1 0.9 --b2 0.98  --eps 10e-9 --dev_out  cfimdb-pt-dev-output.txt --test_out cfimdb-pt-test-output.txt
