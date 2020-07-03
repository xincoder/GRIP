# GRIP

This repository is the code of [GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving](https://arxiv.org/abs/1907.07792) on the Baidu Apollo Trajectory dataset. GRIP++ is an enhanced version of our GRIP ([GRIP: Graph-based Interaction-aware Trajectory Prediction](https://ieeexplore.ieee.org/abstract/document/8917228)).

___
### License
This code is shared only for research purposes, and this cannot be used for any commercial purposes. 

___
### Training 

1. Modify "data_root" in data_process.py and then run the script to preprocess the data. 
``` Bash
$ python data_process.py
```

2. Train the model. We trained the model on a single Nvidia Titan Xp GPU. If your GPU has the same precision, you should get the exact same results. The "training_log.txt" is my training log. If you download the code and run it directly, you should see similar outputs.
``` Bash
$ python main.py

# The following are the first 10 training iterations:
#######################################Train
# |2019-09-20 16:50:43.146035|     Epoch:   0/ 500|	Iteration:    0|	Loss:2.69767785|lr: 0.001|
# |2019-09-20 16:50:43.247776|     Epoch:   0/ 500|	Iteration:    0|	Loss:1.39082634|lr: 0.001|
# |2019-09-20 16:50:43.327926|     Epoch:   0/ 500|	Iteration:    0|	Loss:1.42024708|lr: 0.001|
# |2019-09-20 16:50:43.394658|     Epoch:   0/ 500|	Iteration:    0|	Loss:1.32363927|lr: 0.001|
# |2019-09-20 16:50:43.454833|     Epoch:   0/ 500|	Iteration:    0|	Loss:1.15358388|lr: 0.001|
# |2019-09-20 16:50:43.515517|     Epoch:   0/ 500|	Iteration:    0|	Loss:1.15672326|lr: 0.001|
# |2019-09-20 16:50:43.575027|     Epoch:   0/ 500|	Iteration:    0|	Loss:0.93675584|lr: 0.001|
# |2019-09-20 16:50:43.634769|     Epoch:   0/ 500|	Iteration:    0|	Loss:0.90181452|lr: 0.001|
# |2019-09-20 16:50:43.694374|     Epoch:   0/ 500|	Iteration:    0|	Loss:0.75979233|lr: 0.001|
```
___

### Submission
Once you trained the model, you can test the trained models on the testing subset.

- Our model predicts future locations for all observed objects simultaneously. 
- Using separate models for different types of objects should achieve better performance. 

|Method|Epoch|WSADE|ADEv|ADEp|ADEb|WSFDE|FDEv|FDEp|FDEb|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|TrafficPredict| |8.5881|7.9467|7.1811|12.8805|24.2262|12.7757|11.121|22.7912|
||
|GRIP|Epoch16|1.2632|2.2511|0.718|1.8024|2.3713|4.0863|1.3838|3.4155|
|GRIP|Epoch18|1.2648|2.2515|0.7142|1.8193|2.3677|4.0863|1.3732|3.4274|
|GRIP|Epoch20|1.2721|2.24|0.717|1.8558|2.3921|4.0762|1.3791|3.5318|
||
|GRIP|Combine|1.2588|2.2400|0.7142|1.8024|2.3631|4.0762|1.3732|3.4155|

We use the following way to combine multiple results.

- epoch20 -> 1, 2 (car)
- epoch18 -> 3 (pedestrian)
- epoch16 -> 4 (bike)

___

### Citation
Please cite our papers if you used our code. Thanks.
``` 
@inproceedings{2019itsc_grip,
 author = {Li, Xin and Ying, Xiaowen and Chuah, Mooi Choo},
 booktitle = {2019 IEEE INTELLIGENT TRANSPORTATION SYSTEMS CONFERENCE (ITSC)},
 organization = {IEEE},
 title = {GRIP: Graph-based Interaction-aware Trajectory Prediction},
 year = {2019}
}

@article{li2020gripplus,
  title={GRIP++: Enhanced Graph-based Interaction-aware Trajectory Prediction for Autonomous Driving},
  author={Li, Xin and Ying, Xiaowen and Chuah, Mooi Choo},
  journal={arXiv preprint arXiv:1907.07792},
  year={2020}
}
```
