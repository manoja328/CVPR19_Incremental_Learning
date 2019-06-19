mkdir checkpoint

#Cosine Normalization (Mimic Scores) + Less-forget Constraint (Adaptive Loss Weight) + Margin Ranking Loss + Class Balance Finetune
CUDA_VISIBLE_DEVICES=0 python cbf_class_incremental_cosine_imagenet.py \
    --dataset imagenet --datadir data/imagenet/data/ --num_classes 1000 \
    --nb_cl_fg 100 --nb_cl 100 --nb_protos 20 \
    --resume --rs_ratio 0.0 --imprint_weights \
    --less_forget --lamda 10 --adapt_lamda \
    --random_seed 1993 \
    --mr_loss --dist 0.5 --K 2 --lw_mr 1 \
    --cb_finetune \
    --ckp_prefix cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_cosine_imagenet \
    2>&1 | tee log_cbf_seed_1993_rs_ratio_0.0_all_class_incremental_MR_LFAD_cosine_imagenet_nb_cl_fg_50_nb_cl_10_nb_protos_20.txt
