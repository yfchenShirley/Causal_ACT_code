# Causal-ACT: Causal Structure Learned Action Chunking Transformers

This repo is modified based on Tony Zhao's implementation (https://github.com/tonyzhaozh/act).


    conda activate aloha
    cd causal_ACT/

### For a quick run on the trained Causal-ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir ckpts/example_trained_ACT \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --seed 0 --eval



### Simulated experiments

Step 1: Train Causal-ACT:
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --seed 0

Step 2: Perform intervention by policy execution (ACT):
    
    # Transfer Cube task
    python3 intervention_act_execution.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --lr 1e-5 --seed 0 --num_its 50 --temperature 10

Step 3: Evaluate the policy with the best graph saved in `<ckpt dir>`
    
    # Transfer Cube task
    python3 imitate_episodes.py --task_name sim_transfer_cube_scripted --ckpt_dir <ckpt dir> \
    --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 \
    --num_epochs 2000  --lr 1e-5 --seed 0 --eval


Add ``--onscreen_render`` to see real-time rendering during evaluation.
