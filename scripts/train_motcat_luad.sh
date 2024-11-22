CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /path/to/feature \
--split_dir tcga_brca \
--model_type motcat \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--which_splits 5foldcv \
--apply_sig

CUDA_VISIBLE_DEVICES=0 python main.py \
--data_root_dir /home/cvnlp/WSI_DATA/TCGA_LUAD_feature \
--split_dir tcga_luad \
--model_type pgbf \
--ot_impl pot-uot-l2 \
--ot_reg 0.1 --ot_tau 0.5 \
--modulation OGM_GE \
--modulation_starts 5 --modulation_ends 15 \
--alpha 0.1 \
--which_splits 5foldcv \
--apply_sig