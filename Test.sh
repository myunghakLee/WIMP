python test.py --mode train --dataroot data/argoverse_processed_smoothing --IFC \
--lr 0.0001 --weight-decay 0.0 --non-linearity relu  --use-centerline-features \
--segment-CL-Encoder-Prob --num-mixtures 6 --output-conv --output-prediction \
--gradient-clipping --hidden-key-generator --k-value-threshold 10 \
--scheduler-step-size 60 90 120 150 180  --distributed-backend ddp \
--experiment-name example_test --gpus 2 --batch-size 12 --ckpt_path experiments/Smoothing/Last.ckpt --save_dir ResultsJson/LAST_smoothing_smoothing --save_json

python test.py --mode train --dataroot data/argoverse_processed_smoothing --IFC \
--lr 0.0001 --weight-decay 0.0 --non-linearity relu  --use-centerline-features \
--segment-CL-Encoder-Prob --num-mixtures 6 --output-conv --output-prediction \
--gradient-clipping --hidden-key-generator --k-value-threshold 10 \
--scheduler-step-size 60 90 120 150 180  --distributed-backend ddp \
--experiment-name example_test --gpus 2 --batch-size 12 --ckpt_path experiments/Smoothing/checkpoints/epoch=104.ckpt --save_dir ResultsJson/BEST_smoothing_smoothing --save_json

python test.py --mode train --dataroot data/argoverse_processed --IFC \
--lr 0.0001 --weight-decay 0.0 --non-linearity relu  --use-centerline-features \
--segment-CL-Encoder-Prob --num-mixtures 6 --output-conv --output-prediction \
--gradient-clipping --hidden-key-generator --k-value-threshold 10 \
--scheduler-step-size 60 90 120 150 180  --distributed-backend ddp \
--experiment-name example_test --gpus 2 --batch-size 12 --ckpt_path experiments/Smoothing/Last.ckpt --save_dir ResultsJson/LAST_smoothing_nonsmoothing --save_json

python test.py --mode train --dataroot data/argoverse_processed --IFC \
--lr 0.0001 --weight-decay 0.0 --non-linearity relu  --use-centerline-features \
--segment-CL-Encoder-Prob --num-mixtures 6 --output-conv --output-prediction \
--gradient-clipping --hidden-key-generator --k-value-threshold 10 \
--scheduler-step-size 60 90 120 150 180  --distributed-backend ddp \
--experiment-name example_test --gpus 2 --batch-size 12 --ckpt_path experiments/Smoothing/checkpoints/epoch=104.ckpt --save_dir ResultsJson/BEST_smoothing_nonsmoothing --save_json
