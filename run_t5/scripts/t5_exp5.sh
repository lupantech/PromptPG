python train.py --label exp5 \
--model base \
--batch_size 32 \
--eval_batch_size 32 \
--gpu 0

python inference.py --label exp5 \
--test_split test \
--test_num -1 \
--model base \
--gpu 1 \
--check_point best_model.pth

# 43.52%
# -> ../results/t5/exp5_base.json
python eval.py --result_file t5/exp5_base.json
