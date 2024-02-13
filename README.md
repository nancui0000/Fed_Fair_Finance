# CRAFT Project: Advanced Fairness-aware Federated Learning for Financial Risk Assessment

The objective of this project is to study and design new federated learning techniques to preserve privacy and improve fairness 
in machine learning applications in financial domains. Financial data are generally distributed which means different financial
entities save their own data locally and do not share data for learning purposes. However, machine learning algorithms benefit 
from learning large-scale datasets that cover diverse distributions. Thus, federated learning has become a popular architecture 
for distributed learning where data are saved locally. The goal of federated learning is to be able to harness data without a 
third party ever directly interacting with the data, thus ensuring users’ data remain secure. We will study federated learning 
algorithms for heterogeneous data with fairness constraints. We will design new dynamic and adaptive strategies for parameter sharing 
to handle heterogeneous financial data distributions. To guarantee different clients receive fair predictions, we will develop random
matrix-based aggregation methods to prevent privacy leaking in fairness-constrained federated learning.

[Nan Cui](https://www.linkedin.com/in/nan-cui-997aa9212/), [Yianni Ypsilantis](https://www.linkedin.com/in/ioannis-ypsilantis-42947b254/), [Dominick Varano](https://www.linkedin.com/in/dominick-varano-697a38241/), [Yue Ning](https://yue-ning.github.io/)


## Requirements

The code has been successfully tested in the following environment. (For older versions, you may need to modify the code.)

- imbalanced_learn==0.11.0
- matplotlib==3.6.2
- numpy==1.23.5
- pandas==1.5.2
- scikit_learn==1.3.0
- torch==1.13.1
- torchvision==0.14.1
- tqdm==4.64.1


## Installations

```bash
pip install -r requirements.txt
```

## Usage

For the adult dataset:

```bash
python src/federated_main.py --model=simple_MLP --dataset=adult --gpu=0 --optimizer=adam --epochs=50 --local_ep=20 --lr=0.001 --sens_drop_flag=1 --aggregate_method=fair --imbalance_method=adasyn --iid=0 --male_to_female_ratio=1 --num_users=100 --loss_type=bce --beta=0.6 --tau=1 --fair_regulization=1
```

## Results

For the adult dataset:

Results after 50 global rounds of training:

### Overall and Demographic-Specific Performance

| Metric | Overall | Females | Males | Income > 50k | Income < 50k |
|--------|---------|---------|-------|--------------|--------------|
| **Accuracy (%)** | 76.03 | 74.15 | 79.84 | 76.51 | 75.88 |
| **Recall** | 0.77 | 0.76 | 0.82 | 0.77 | 0.76 |
| **F1 Score** | 0.60 | 0.64 | 0.47 | 0.87 | 0.86 |
| **AUC** | 0.76 | 0.75 | 0.81 | N/A | N/A |
| **Fairness Metrics** | | | | | |
| **Test Parity (%)** | 13.98 | - | - | - | - |
| **Test Equality (%)** | 6.53 | - | - | - | - |

## Cite

Please cite our project if you find this code useful for your research:

```
N. Cui, Y. Ypsilantis, D. Varano and Y. Ning, "Advanced Fairness-aware Federated Learning for Financial Risk Assessment."
```
