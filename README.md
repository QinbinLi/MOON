# Model-Contrastive Federated Learning
This is the code for paper [Model-Contrastive Federated Learning](https://arxiv.org/pdf/2103.16257.pdf).

**Abstract**: Federated learning enables multiple parties to collaboratively train a machine learning model without communicating their local data. A key challenge in federated learning is to handle the heterogeneity of local data distribution across parties. Although many studies have been proposed to address this challenge, we find that they fail to achieve high performance in image datasets with deep learning models. In this paper, we propose MOON: model-contrastive federated learning. MOON is a simple and effective federated learning framework. The key idea of MOON is to utilize the similarity between model representations to correct the local training of individual parties, i.e., conducting contrastive learning in model-level. Our extensive experiments show that MOON significantly outperforms the other state-of-the-art federated learning algorithms on various image classification tasks.

## Dependencies
* PyTorch >= 1.0.0
* torchvision >= 0.2.1
* scikit-learn >= 0.23.1



## Parameters

| Parameter                      | Description                                 |
| ----------------------------- | ---------------------------------------- |
| `model`                     | The model architecture. Options: `simple-cnn`, `resnet50` .|
| `alg` | The training algorithm. Options: `moon`, `fedavg`, `fedprox`, `local_training` |
| `dataset`      | Dataset to use. Options: `cifar10`. `cifar100`, `tinyimagenet`|
| `lr` | Learning rate. |
| `batch-size` | Batch size. |
| `epochs` | Number of local epochs. |
| `n_parties` | Number of parties. |
| `sample_fraction` | the fraction of parties to be sampled in each round. |
| `comm_round`    | Number of communication rounds. |
| `partition` | The partition approach. Options: `noniid`, `iid`. |
| `beta` | The concentration parameter of the Dirichlet distribution for non-IID partition. |
| `mu` | The parameter for MOON and FedProx. |
| `temperature` | The temperature parameter for MOON. |
| `out_dim` | The output dimension of the projection head. |
| `datadir` | The path of the dataset. |
| `logdir` | The path to store the logs. |
| `device` | Specify the device to run the program. |
| `seed` | The initial seed. |


## Usage

Here is an example to run MOON on CIFAR-10 with a simple CNN:
```
python main.py --dataset=cifar10 \
    --model=simple-cnn \
    --alg=moon \
    --lr=0.01 \
    --mu=5 \
    --epochs=10 \
    --comm_round=100 \
    --n_parties=10 \
    --partition=noniid \
    --beta=0.5 \
    --logdir='./logs/' \
    --datadir='./data/' \
```

## Tiny-ImageNet
You can download Tiny-ImageNet [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip). Then, you can follow the [instructions](https://github.com/AI-secure/DBA/blob/master/utils/tinyimagenet_reformat.py) to reformat the validation folder.

## Hyperparameters
If you use the same setting as our papers, you can simply adopt the hyperparameters reported in our paper. If you try a setting different from our paper, please tune the hyperparameters of MOON. You may tune mu from \{0.001, 0.01, 0.1, 1, 5, 10\}. If you have sufficient computing resources, you may also tune temperature from \{0.1, 0.5, 1.0\} and the output dimension of projection head from \{64, 128, 256\}. 



## Citation

Please cite our paper if you find this code useful for your research.
```
@inproceedings{li2021model,
      title={Model-Contrastive Federated Learning}, 
      author={Qinbin Li and Bingsheng He and Dawn Song},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      year={2021},
}
```
