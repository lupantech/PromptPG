python train.py --label exp4 \
--model small \
--batch_size 32 \
--eval_batch_size 32 \
--gpu 1

python inference.py --label exp4 \
--test_split test \
--test_num -1 \
--model small \
--gpu 1 \
--check_point best_model.pth

# 29.79%
# -> ../results/t5/exp4_small.json
python eval.py --result_file t5/exp4_small.json
