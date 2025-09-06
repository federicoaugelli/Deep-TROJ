# Deep-TROJ
Official code repository of Deep-TROJ (CVPR 2024)

Download all weights (post attack optimization) from this [link](https://drive.google.com/file/d/17pS3K1w7mKFdO5q0Mh4CVvOH7ZEIL2nP/view?usp=drive_link)

1) Carry out attack optimization on CNN models:
   ```
   python attack_optimization_new.py --dataset=cifar10 --rounds=10 --n_blocks=5 --device=cuda:0 --exp_path=results_n_blocks_5_new --mixed_precision
   ```
   
2) Carry out attack optimization on Transformer model (DeiT-S):
   ```
   python attack_transformer_new.py --dataset=imagenet --rounds=5 --n_blocks=5 --device=cuda:0 --exp_path=results_n_blocks_5_new --mixed_precision
   ```
   
3) Evaluate attack performance on CNN model after optimization
   ```
   python evaluate_attack.py --dataset=cifar10 --exp_path=results_n_blocks_5_new --device=cuda:0 --mixed_precision
   ```

4) Evaluate attack performance on Transformer model (DeiT-S) after optimization
   ```
   python evaluate_transformer.py --dataset=imagenet --exp_path=results_n_blocks_5_new --device=cuda:0 --mixed_precision
   ```

## defense methods

### Autoencoder (tested)
preprocess input through autoencoder. Train autoencoder with train set. Inference time every input forced to enter the distribution of train data

### Gaussian filter (x)
trojan are sensible to pixel-level perturbations. Apply a gaussian filter to remove perturbations before entering NN. Lower accuracy

### GAN (x)
Gradient-weighted class activation mapping (GradCAM). Generate heatmap. remove and reconstruct

### retrain (tested)
Label trigger correctly and retrain model

### Random noise (tested)
use random noise (max entropy staircase approximation) to evaluate distribution

### Pruning (x)

### ORAM (oblivious RAM) (x)
