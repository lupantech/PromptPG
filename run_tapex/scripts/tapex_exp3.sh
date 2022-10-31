## Pre-trained
python inference.py --label exp3 \
--test_split test \
--test_num -1 \
--model tapex-large \
--gpu 0

# 18.59%
python eval.py --result_file tapex/exp3_tapex-large.json