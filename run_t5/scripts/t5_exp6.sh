python train.py --label exp6 \
--model large \
--batch_size 8 \
--eval_batch_size 32 \
--gpu 0

python inference.py --label exp6 \
--test_split test \
--test_num -1 \
--model large \
--gpu 0 \
--check_point best_model.pth

# 57.35%
# -> ../results/t5/exp6_large.json
python eval.py --result_file t5/exp6_large.json
