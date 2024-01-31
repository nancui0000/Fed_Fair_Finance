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

|---- Avg Train Accuracy: 78.69%

-----------------------------------------------

|---- Test Accuracy: 76.03%

|---- Test Recall: 0.77

|---- Test F1: 0.60

|---- Test AUC: 0.76

|---- Test Parity: 13.98%

|---- Test Equality: 6.53%

-----------------------------------------------

 Total Run Time: 47.2392
 
|---- Test over Females Accuracy: 74.15%

|---- Test over Females Recall: 0.76

|---- Test over Females F1: 0.64

|---- Test over Females AUC: 0.75

-----------------------------------------------

|---- Test over Males Accuracy: 79.84%

|---- Test over Males Recall: 0.82

|---- Test over Males F1: 0.47

|---- Test over Males AUC: 0.81

-----------------------------------------------

|---- Test over Income > 50k Accuracy: 76.51%

|---- Test over Income > 50k Recall: 0.77

|---- Test over Income > 50k F1: 0.87

|---- Test over Income > 50k AUC: nan

-----------------------------------------------

|---- Test over Income < 50k Accuracy: 75.88%

|---- Test over Income < 50k Recall: 0.76

|---- Test over Income < 50k F1: 0.86

|---- Test over Income < 50k AUC: nan

-----------------------------------------------

## Cite

Please cite our project if you find this code useful for your research:

```
N. Cui, Y. Ypsilantis, D. Varano and Y. Ning, "Advanced Fairness-aware Federated Learning for Financial Risk Assessment."
```
