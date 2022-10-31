python inference.py --label exp3_pretrained_unifiedqa \
--test_split test \
--test_num -1 \
--model large \
--gpu 0 

# 15.96%
python eval.py --result_file t5/exp3_pretrained_unifiedqa_large.json
