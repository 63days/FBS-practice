# Pytorch Implementation of Feature Boosting and Suppresion
Pytorch implementation of [Dynamic Channel Pruning: Feature Boosting and Suppression](https://arxiv.org/abs/1810.05331).

![image](https://user-images.githubusercontent.com/37788686/98621488-b075e300-234a-11eb-9efe-97eac1c97efe.png)

## Results
|   | baseline | d=1.0 | d=0.9 | d=0.8 | d=0.7 | d=0.6 | d=0.5 | d=0.4 | d=0.3 |
| - | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|Acc | 87.3% | 86.31% | 86.43% | 86.65% | 86.65% | 85.69% | 85.45% | 83.91% | 82.51% |
| MACs | 1.74 | 1.67 | 1.38 | 1.11 | 0.84 | 0.62 | 0.44 | 0.28 | 0.16 |

* d: rate of channel pruning
* MACs: multiply-accumulate operations
![image](https://user-images.githubusercontent.com/37788686/98621800-6ccfa900-234b-11eb-911e-3ca64d0c27bd.png)
