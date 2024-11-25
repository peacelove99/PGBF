CUDA_VISIBLE_DEVICES=1 python main.py \
--data_root_dir /home/cvnlp/WSI_DATA/TCGA_LUAD_feature \
--split_dir tcga_luad \
--model_type motcat \
--bs_micro 256 \
--ot_impl pot-uot-l2 \
--ot_reg 0.05 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /home/cvnlp/WSI_DATA/TCGA_LUAD_feature \
--split_dir tcga_luad \
--model_type pgbf \
--ot_impl pot-uot-l2 \
--ot_reg 0.05 --ot_tau 0.5 \
--modulation OGM_GE \
--modulation_starts 0 --modulation_ends 40 \
--alpha 0.1 \
--max_epochs 40 \
--which_splits 5foldcv \
--apply_sig
