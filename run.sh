#CUDA_VISIBLE_DEVICES=4 python train.py --output_directory=models/apex --log_directory=logs/apex
#CUDA_VISIBLE_DEVICES=5 python train.py --output_directory=models --log_directory=logs

#name=mellotron_pretrained
#CUDA_VISIBLE_DEVICES=7 python -m multiproc train.py \
#    --output_directory=models/$name --log_directory=logs/$name \
#    -c models/mellotron_libritts.pt --warm_start


#name=mellotron_nof0
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m multiproc train.py --hparams=distributed_run=True \
#    --output_directory=models/$name --log_directory=logs/$name \
#    -c models/mellotron_libritts.pt --warm_start

#name=mellotron_pretrained_torch11
#CUDA_VISIBLE_DEVICES=3 python train.py \
#    --hparams=distributed_run=False \
#    --output_directory=models/$name --log_directory=logs/$name \
#    -c models/mellotron_nof0/checkpoint_7000
#    -c models/mellotron_libritts.pt --warm_start

#name=mellotron_nof0
#CUDA_VISIBLE_DEVICES=4,5,6,7 python -m multiproc train.py --hparams=distributed_run=True \
#    --output_directory=models/$name --log_directory=logs/$name \
#    -c models/mellotron_nof0/checkpoint_15500


name=tst_tacotron2_ctxgru
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m multiproc train.py --hparams=distributed_run=True \
    --output_directory=models/$name --log_directory=logs/$name \
    -c models/mellotron_libritts.pt --warm_start

