# Practice FBS Implementation
An pytorch implementation of Feature Boosting and Supression(FBS).

## Introduction
State-of-the-art vision and image-based tasks are all built upon deep _convolutional neural networks_(CNNs). However, CNN architectures usually require considerable memory utilization, bandwidth and compute requirements. The formidable amount of computational resources used by CNNs present a great challenge in the deployment of CNNs in both cost-sensitive cloud services and low-powered edge computing applications. One common approach to reduce the memory is to prune over-parameterized CNNs.
- static pruning(e.g. channel pruning)
- dynamic pruning(skip unnecessary computation according to the input)
### Static Pruning vs Dynamic Pruning
Static Pruning permanently remove whole structure of weights. So it reduce model size efficiently but capabilities of CNNs are permanently lost. In other hand, Dynamic pruning skips unnecessary computation according to the input. So it preserves full CNN structure while computation speed is faster. But it has limitation in model size reduction.
## FBS(Feature Boosting and Suppression)
_feature boosting and suppression_(FBS) dynamically amplify and supress output channels computed by the convolutional layer to reduce the memory. FBS introduces tiny auxiliary connections to existing convolutional layers.
![image](https://user-images.githubusercontent.com/37788686/98621488-b075e300-234a-11eb-9efe-97eac1c97efe.png)

## Results
|   | baseline | d=1.0 | d=0.9 | d=0.8 | d=0.7 | d=0.6 | d=0.5 | d=0.4 | d=0.3 |
| - | -------- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|Acc | 87.3% | 86.31% | 86.43% | 86.65% | 86.65% | 85.69% | 85.45% | 83.91% | 82.51% |
| MACs | 1.74 | 1.67 | 1.38 | 1.11 | 0.84 | 0.62 | 0.44 | 0.28 | 0.16 |

* d: rate of channel pruning
* MACs: multiply-accumulate operations
![image](https://user-images.githubusercontent.com/37788686/98621800-6ccfa900-234b-11eb-911e-3ca64d0c27bd.png)
