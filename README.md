

<h1 align="center">Beyond The FRONTIER</h1>

<p align="center">
    <a href="https://www.tensorflow.org/">
        <img src="https://img.shields.io/badge/Tensorflow-1.13-green" alt="Vue2.0">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Author-JialiCui-blueviolet" alt="Author">
    </a>
    <a href="https://github.com/CuiJiali-CV/">
        <img src="https://img.shields.io/badge/Email-cuijiali961224@gmail.com-blueviolet" alt="Author">
    </a>
    <a href="https://www.stevens.edu/">
        <img src="https://img.shields.io/badge/College-SIT-green" alt="Vue2.0">
    </a>
</p>


[Paper HyperNet](https://arxiv.org/pdf/1905.02898.pdf)
<br /><br />
## BackGround

* This is a good test for using HyperNet idea on VAE

<br />

## Quick Start
* Run train.py and the result will be stored in output
* The hyper params are already set up in train.py.
* The parms used for HyperNet are set up in params.py. If you have a better computer, feel free to try different params.
* The number of trainning images can be set up in loadData.py. Just simply change num to any number of images you want to train

<br /><br />
## Version of Installment
#### Tensorflow 1.13.1
#### Numpy 1.18.2
#### Python 3.6.9  

## Resulte
<br />

### Generation
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/Generation.png)
<br />
### Reconstruction
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/Reconstruction.png) 
<br />
### Test of HyperNet
#### In order to test the HyperNet, two experiment are designed as following
##### Experiment one : Input two different weights and one noise
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/diff_weights_same_z1.png)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/diff_weights_same_z2.png)
##### Experiment two : Input two different noise and one weight
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/diff_z_same_weight1.png)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/diff_z_same_weight2.png)
##### Experiment three : Input two different weight and one img
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/diff_weights_same_img1.png)
 ![Image text](https://github.com/CuiJiali-CV/HyperNet-VAE/raw/master/img/diff_weights_same_img2.png)
<br /><br />
## Author

```javascript
var iD = {
  name  : "Jiali Cui",
  
  bachelor: "Harbin Institue of Technology",
  master : "Stevens Institute of Technology",
  
  Interested: "CV, ML"
}
```
