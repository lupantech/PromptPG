## Training
python learn_policy.py \
--label exp1 \
--ckpt_root ../checkpoints \
--shot_number 2 \
--prompt_format TQ-SA \
--seed 2 \
--model_config bert-base-uncased \
--train_number 160 \
--cand_number 20 \
--lr 0.001 \
--epochs 20 \
--embedding_size 128 \
--batch_size 20 \
--gpu 0


# ## Inference w/ learned checkpoint (dev1k)
# python run_gpt3.py \
# --label exp1 \
# --ckpt_root ../checkpoints \
# --model gpt3_rl \
# --test_split dev1k \
# --test_number -1 \
# --shot_number 2 \
# --prompt_format TQ-SA \
# --seed 2 \
# --cand_number 20 \
# --embedding_size 128 \
# --model_config bert-base-uncased \
# --ckpt exp1/ckpt_best_reward.pt \
# --gpu 0



## Inference w/ learned checkpoint (test)
python run_gpt3.py \
--label exp1 \
--ckpt_root ../checkpoints \
--model gpt3_rl \
--test_split test \
--test_number -1 \
--shot_number 2 \
--prompt_format TQ-SA \
--seed 2 \
--cand_number 20 \
--embedding_size 128 \
--model_config bert-base-uncased \
--ckpt exp1/ckpt_best_reward.pt \
--gpu 0
