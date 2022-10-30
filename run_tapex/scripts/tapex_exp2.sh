## Fine-tuned
python train.py --label exp2 \
--model tapex-base \
--batch_size 16 \
--eval_batch_size 16 \
--gpu 1

## Inference
python inference.py --label exp2 \
--test_split test \
--test_num -1 \
--model tapex-base \
--gpu 0 \
--check_point best_model.pth

# 48.27%
python eval.py --result_file tapex/exp2_tapex-base.json
