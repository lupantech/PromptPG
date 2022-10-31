python inference.py --label exp1_pretrained_unifiedqa \
--test_split test \
--test_num -1 \
--model small \
--gpu 0 

# 12.18%
python eval.py --result_file t5/exp1_pretrained_unifiedqa_small.json
