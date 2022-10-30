python inference.py --label exp2_pretrained_unifiedqa \
--test_split test \
--test_num -1 \
--model base \
--gpu 0 

# 14.56%
python eval.py --result_file t5/exp2_pretrained_unifiedqa_base.json
