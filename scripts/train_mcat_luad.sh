CUDA_VISIBLE_DEVICES=2 python main.py \
--data_root_dir /home/cvnlp/WSI_DATA/TCGA_LUAD_feature \
--split_dir tcga_luad \
--model_type mcat \
--max_epochs 40 \
--which_splits 5foldcv \
--apply_sig