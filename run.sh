python main_CFT_stage2.py --dataset CREMAD --num_frame 3 --load_path path_model1 --load_path_other path_model2 --temperature 1 --train


python main_CFT.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.2
python main_CFT.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.3
python main_CFT.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.4

python main_CFT_stage2.py --dataset CREMAD --num_frame 3 --load_path ckpt/CFT-audio/model-CREMAD-bsz16-lr0.001-align1/epoch-149.pt --load_path_other ckpt/CFT-visual/model-CREMAD-bsz16-lr0.001-align1/epoch-149.pt --temperature 1 --train --var 0.1
python main_CFT_stage2.py --dataset CREMAD --num_frame 3 --load_path ckpt/CFT-audio/model-CREMAD-bsz16-lr0.001-align1/epoch.pt --load_path_other ckpt/CFT-visual/model-CREMAD-bsz16-lr0.001-align1/epoch.pt --temperature 1 --train --var 0.3

python main_joint_training.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.1
python main_joint_training.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.2
python main_joint_training.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.3
python main_joint_training.py --dataset CREMAD --num_frame 3 --temperature 50 --train --var 0.4



python main_CFT_stage2.py --dataset CREMAD --num_frame 3 --load_path ckpt/CFT-audio/model-CREMAD-bsz16-lr0.001-align1-var0.4/epoch.pt --load_path_other ckpt/CFT-visual/model-CREMAD-bsz16-lr0.001-align1-var0.4/epoch.pt --temperature 1 --train --var 0.4

