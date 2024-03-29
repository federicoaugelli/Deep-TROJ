# Deep-TROJ
Official code repository of Deep-TROJ (CVPR 2024)

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

