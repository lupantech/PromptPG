## Fine-tuned
python train.py --label exp4 \
--model tapex-large \
--batch_size 8 \
--eval_batch_size 8 \
--gpu 0 \
--lr 2e-5

## Inference
python inference.py --label exp4 \
--test_split test \
--test_num -1 \
--model tapex-large \
--gpu 0 \
--check_point best_model.pth

# 58.52%
python eval.py --result_file tapex/exp4_tapex-large.json
