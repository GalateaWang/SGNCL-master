## Training & Evaluation

### Pre-training:

```
cd ./transfer
python pre_train.py --datasets zinc_standard_agent --save_model_path ./models_sgncl/sgngcl --mode1 0 --mode2 1
```

### Finetuning:

```python
cd ./transfer
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 0
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 1
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 2
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 3
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 4
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 5
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 6
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 7
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 8
python finetune.py --dataset bbbp --input_model_file models_sgncl/sgngcl_seed0_40 --runseed 9
```