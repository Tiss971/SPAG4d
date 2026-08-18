# IMAGE TEST
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/plan_usine_1.jpg /raid/mb273924/_DATASETS/uptale/debug_images/da360.ply --stride=8  --generator "da360"
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/plan_usine_1.jpg /raid/mb273924/_DATASETS/uptale/debug_images/sharp360.ply --stride=8  --generator "sharp360"
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/plan_usine_1.jpg /raid/mb273924/_DATASETS/uptale/debug_images/unisharp360.ply --stride=8  --generator "unisharp360"
# VIDEO TEST
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/accident_electrique_02.mp4 /raid/mb273924/_DATASETS/uptale/_debug_bchmk2/accident_electrique_02/accident_electrique_02.ply --stride=8 --alignement-mask="sam" --alignement-method="lstsq"  --skip-step 1 --freeze-bg
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/MattSwift.mp4 /raid/mb273924/_DATASETS/uptale/_debug_bchmk2/MattSwift/MattSwift.ply --stride=8 --alignement-mask="sam" --alignement-method="lstsq"  --skip-step 1 --freeze-bg
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/tissa.mp4 /raid/mb273924/_DATASETS/uptale/_debug_bchmk2/tissa/tissa.ply --stride=8 --alignement-mask="sam" --alignement-method="lstsq"  --skip-step 1 --freeze-bg
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/tissb.mp4 /raid/mb273924/_DATASETS/uptale/_debug_bchmk2/tissb/tissb.ply --stride=8 --alignement-mask="sam" --alignement-method="lstsq"  --skip-step 1 --freeze-bg
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/tissc.mp4 /raid/mb273924/_DATASETS/uptale/_debug_bchmk2/tissc/tissc.ply --stride=8 --alignement-mask="sam" --alignement-method="lstsq"  --skip-step 1 --freeze-bg
# python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/accident_electrique_fast5.mp4 /raid/mb273924/FreeTimeGsVanilla/_data/accident_electrique_fast5_raw --stride=4 --skip-step 1 --freeze-bg --depth-correction bglock
python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/accident_electrique_fast5.mp4 /raid/mb273924/FreeTimeGsVanilla/_data/accident_electrique_fast5_likefreetime_loweroutlierprune --stride=4 --skip-step 1 --freeze-bg --depth-correction bglock --outlier-pruning 0.1 --grazing-angle 85.0 --sparse-pruning 0.1
python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/accident_electrique_fast5.mp4 /raid/mb273924/FreeTimeGsVanilla/_data/accident_electrique_fast5_onlyprune --stride=4 --skip-step 1 --freeze-bg --depth-correction bglock --outlier-pruning 0.3 
python -m spag4d convert /raid/mb273924/_DATASETS/uptale/data/accident_electrique_fast5.mp4 /raid/mb273924/FreeTimeGsVanilla/_data/accident_electrique_fast5_onlygrazing_and_sparse --stride=4 --skip-step 1 --freeze-bg --depth-correction bglock --grazing-angle 85.0 --sparse-pruning 0.1

