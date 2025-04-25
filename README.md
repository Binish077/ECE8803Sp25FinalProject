# ECE 8803 Spring 2025 Final Project - OLIVES Biomarker Detection


***

This work was done based on research by [Omni Lab for Intelligent Visual Engineering and Science (OLIVES) @ Georgia Tech](https://ghassanalregib.info/). 
This project is based on the competition from [OLIVES](https://proceedings.neurips.cc/paper_files/paper/2022/hash/3be60b4a739b95a07a944a1a2c41e05e-Abstract-Datasets_and_Benchmarks.html) dataset published at NeurIPS 2022.

***

## Data

Training_Biomarker_Data.csv : Biomarker labels in training set.

Training_Unlabeled_Clinical_Data.xlsx : Provides the clinical labels for all the data without biomarker information.

test_set_submission_template.csv : This provides the structure by which all submissions should be organized. 
This includes the image path and the associated 6 biomarkers.

PRIME_FULL and TREX DME are the training sets.

RECOVERY is the test set. The ground truth biomarker labels are held out, but the images and clinical data are provided.

## Starter Code Usage

```bash
python train.py --batch_size 128 --model 'resnet18' --dataset 'OLIVES' --epochs 1 --device 'cuda:0' --train_image_path '' --test_image_path '' --test_csv_path './csv_dir/test_set_submission_template.csv' --train_csv_path './csv_dir/Training_Biomarker_Data.csv'
```

## Model Options
The model that is trained can be changed using the `--model` argument:
- EnhancedResnet

## To test against the RECOVERY dataset
To evaluate the model against the RECOVERY dataset:
- Uncomment line 127 in `train.py`:  
  ```python
  # train_loader, test_loader = set_loader(opt)
  ```
- Comment out line 130:  
  ```python
  train_loader, test_loader = set_loader_val(opt)
  ```
- Comment out line 147:  
  ```python
  evaluate_model(test_loader, model, opt)
  ```
- Uncomment line 150:  
  ```python
  # submission_generate(test_loader, model, opt)
  ```
- Ensure the dataset is loaded using the `set_loader` function from `utils.py`, **not** `set_loader_val`.

- Vice versa can be done to use a 80/20 train-test split to evaluate the model instead.

### Acknowledgements

This work was done in collaboration with the [Retina Consultants of Texas](https://www.retinaconsultantstexas.com/).
This codebase utilized was partly constructed with code from the [Supervised Contrastive Learning](https://github.com/HobbitLong/SupContrast) Github.
The datasets used in this project are from the work provided by Prabhushankar, M., Kokilepersaud, K., Logan, Y. Y., Trejo Corona, S., AlRegib, G., \& Wykoff, C. (2022). Olives dataset: Ophthalmic labels for investigating visual eye semantics. Advances in Neural Information Processing Systems, 35, 9201-9216.
This project was based on the [OLIVES] 2023 IEEE SPS Video and Image Processing Cup (VIP Cup) on Ophthalmic Biomarker Detection competition. We thank the organizers of the competition for providing resources to innovate and explore the field of medical image deep learning.
