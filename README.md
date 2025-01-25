This project is used for vulnerability detection of smart contracts, which mainly involves three types of reentry vulnerabilities, timestamp dependency vulnerabilities and integer overflow vulnerabilities.
The versions of the main packages involved are as follows:

```
Python --3.10
Pytorch--2.5.0
CUDA--12.4（choose for oneself）
```


```
Create -n cegt python==3.10
Conda activate cegt
```

**DATASET:**
We use the dataset from [link](https://github.com/Messi-Q), including the reentry vulnerability and timestamp dependency vulnerability datasets
Integer overflow vulnerabilities from [link](https://github.com/Messi-Q)

The code structure is as follows:
**trans_test.py** 

> mainly involves reading and partitioning of the dataset, training and testing of the model, as well as the calculation of evaluation performance metrics

**models.py** and **gcnt.py** 

> include the specific implementation of the model, and the specific functions are realized by calling layer.py.

**parser1.py** 

> the setting of each parameter including hyperparameters, including --dataset to select dataset, lr, dropout, n_hidden, epoch, batch_size.

RUN:

1. Modify the configuration parameters
Select the dataset in the parser1.py file, or use the command to pass the parameter to select the dataset; modify the value of the hyperparameter in the file or use the command to pass the parameter to set the hyperparameter
2. Set the path to save the training log
Save the input results to the log
3. Run the code for training and testing
Run trans_test.py, get the results of training and testing
such as python trans_test.py
