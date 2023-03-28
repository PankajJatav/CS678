python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-1 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-1    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-3 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-3    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-7 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-7    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 1 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 1    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 5 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 5    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 7 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 7    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 8   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 16 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 16   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 24 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 24   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 32 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 32   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir

python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 1 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output2'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir

python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 2 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 2   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir

python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 4 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 4   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir

python src/run_squad.py --model_type bert --model_name_or_path=bert-base-multilingual-uncased --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 6 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 6   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type bert --model_name_or_path=google/bert_uncased_L-12_H-768_A-12 --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type bert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type xlm --model_name_or_path=deepset/xlm-roberta-large-squad2 --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type xlm  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type roberta --model_name_or_path=deepset/roberta-base-squad2 --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type roberta  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type distilbert --model_name_or_path=distilbert-base-cased-distilled-squad --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type distilbert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type camembert --model_name_or_path=etalab-ia/camembert-base-squadFR-fquad-piaf --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type camembert  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir


python src/run_squad.py --model_type longformer --model_name_or_path=valhalla/longformer-base-4096-finetuned-squadv1 --do_train --do_lower_case --train_file 'data/dialqa-train.json' --per_gpu_train_batch_size 8 --per_gpu_eval_batch_size 24 --learning_rate 3e-5 --num_train_epochs 3 --max_seq_length 184 --doc_stride 128 --output_dir '/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output/' --overwrite_cache --overwrite_output_dir
python src/run_squad.py   --model_type longformer  --model_name_or_path='/media/pankaj/77b908cd-def5-4c1e-9be0-0a0a117da625/home/pankaj/Study/NLP/train_cache_output'     --do_eval        --do_lower_case         --predict_file 'data/dialqa-dev-aug.json'       --per_gpu_train_batch_size 1   --per_gpu_eval_batch_size 16    --learning_rate 3e-5    --num_train_epochs 3    --max_seq_length 384     --doc_stride 128        --output_dir 'outputs/aug-mbert' --overwrite_output_dir