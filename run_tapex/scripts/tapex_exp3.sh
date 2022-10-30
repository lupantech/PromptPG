## Pre-trained
python inference.py --label exp3 \
--test_split test \
--test_num -1 \
--model tapex-base \
--gpu 0

# 15.73%
python eval.py --result_file tapex/exp3_tapex-base.json

