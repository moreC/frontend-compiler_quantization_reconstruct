This repo provides various tools for neural network front-end compiler, quantization and framework converter among tensorflow, pytorch and mxnet

An example of a resnet50 model from mxnet is used to demo the above mentioned tools 

---
### 1. Get Environment Docker 

Pull Image from Dockerhub

```
docker pull xhxian/frontendcompiler:0.0.1

docker run -ti -v /path/to/your/file/directory:/directory_name xhxian/frontendcompiler:0.0.1 /bin/bash
```

---
###  2. Frontend compiler usage

### Directory list
```
|-- configs
|-- examples
|-- projects
|   |-- fast_prune
|   |   -- fast_pruning
|   |       |-- core
|   |       |-- data
|   |       |-- optimizer
|   |       -- pruning
|   |-- tensorflow
|   |   -- cifar10
|   |       -- include
|   -- torch
|       -- cifar10
|-- src
|   |-- compiler
|   |-- dataset
|   |-- np_ops
|   |-- postprocess
|   |   -- tf
|   |-- quantization
|   |-- reconstructor
|   -- utils
-- tools
    | -- images
```
- configs: 
```
contain a few configuration files for tools
```


- examples
```
|-- README.md
|-- mxnet_eval.py
|-- test_calibration.py
|-- test_complier.py
|-- test_gluoncv_centernet_rt.py
|-- test_imagenet_quantize_acc.py
|-- test_model_complexity.py
|-- test_pytorch_reconstructor.py
|-- test_quantize_error.py
|-- test_tensorflow_recostructor.py
|-- test_tensorflow_recostructor_train.py
|-- test_tvm_recontructor.py
```

More details can be found in [Frontend Compiler Example Documentation](examples/Frontend_Compiler_Examples.md)

- projects

projects show two demos: tensorflow reconstruct, pytorch reconstruct
```
|-- tensorflow
|   |-- cifar10
|       |-- include
|-- torch
    |-- cifar10
```

- src

src include the source codes for compiler，quantization，reconstructor，utils. Detailed document can [Frontend Compiler Core Documentation](examples/Frontend_Compiler_Core.md)

```
|-- compiler
|-- dataset
|-- np_ops
|-- postprocess
|   |-- tf
|-- quantization
|-- reconstructor
|-- utils
```
---
### 3. Step-by-step  expample

### 3.1: get original model

we use gluoncv resnet50v1b (top1=77.67 on imagenet) as an example here

### 3.2: prune the dense network using the [sparse tools](https://github.com/jiacliu09/PruningTools) provided by Moffett AI 

| model        | sparse rate | acc top1 | input size |
| ------------ | ----------- | -------- | ----------- |
| resnet50 v1b | 0           | 77.7    | 224 x 224 | 
| resnet50 v1b | 93.75       | 74.1    | 224 x 224 | 

The checkpoint can be saved as follows:
```
resnet50v1b/
├── sp93.75_224x224
│   ├── resnet50v1b_sparse_93.75-0199.params
│   └── resnet50v1b_sparse_93.75-symbol.json
└── original
    ├── resnet50v1b-0000.params
    └── resnet50v1b-symbol.json
```

### 3.3: frontend model compiler test
compile the sparse mxnet model into moffett IR

```python ./examples/test_complier.py mxnet resnet50v1b 0 ```

Moffett IR will be saved in the folder `./moffett_ir`
```
├── IR_for_reconstruct_graph.json
├── IR_for_reconstruct_graph.png
├── IR_for_reconstruct_params.npz
├── IR_fused_for_CModel_graph.json
├── IR_fused_for_CModel_graph.png
└── IR_fused_for_CModel_params.npz
```

### step 4: model complexity
For the given model, calcualte the non-zeros, sparsity, dense flops, sparse flops
```
python examples/test_model_complexity.py 
        --graph-json moffett_ir/IR_for_reconstruct_graph.json
        --params-file moffett_ir/IR_for_reconstruct_params.npz
```
nnz: 3522685, sparsity: 0.8618712639560533, dense_flops: 3948251920.0, sparse_flops: 357813835.0

### step 5: reconstructor test 
使用reconstructor之前最好使用训练框架生成一个input和result测资（保存为npy），以方便验证reconstructor是否正确，假设目前已经生成了input(input.npy)和result（result.npy）,需要注意一点目前IR_for_reconstrcut和IR_fused_for_CModel都能reconstruct，但是面向的目标有所不一样，如果需要训练，则不能用IR_fused_for_CModel
In pytorch:
```
python examples/test_pytorch_reconstructor.py 
        --graph moffett_ir/IR_for_reconstruct_graph.json 
        --params moffett_ir/IR_for_reconstruct_params.npz 
        --input_npy input.npy 
        --result_npy result.npy
```
最终产生一个distance的结果，如果distance > 1e-3，则反馈至@zhilong，反之则补足表格

| model        | orig model top1 | training date | input_shape | compiler pass | torch rt |
| ------------ | --------------- | ------------- | ----------- | ------------- | -------- |
| resnet50 v1b | 77.67           | 20200712      | 224x224     | yes           | yes      |
| resnet50 v1b | 77.67           | 20200712      | 224x224     | yes           | yes      |

In tensorflow:
```
python examples/test_tensorflow_reconstructor.py 
        --graph moffett_ir/IR_for_reconstruct_graph.json 
        --params moffett_ir/IR_for_reconstruct_params.npz 
        --input_npy input.npy 
        --result_npy result.npy 
        --save_path tf_reconstruct.pb
```
最终会产生一个distance,同时会产生对应的frozen pb，补足表格

| model        | training date | input_shape | compiler pass | torch rt | tf rt |
| ------------ | ------------- | ----------- | ------------- | -------- | ----- |
| resnet50 v1b | 20200712      | 224x224     | yes           | yes      | yes   |
| resnet50 v1b | 20200712      | 224x224     | yes           | yes      | yes   |

### step6: quantization test
config file documentation:
```
MODEL:
    graph: './models/resnet50_v1b/IR_fused_for_CModel_graph.json'
    params: './models/resnet50_v1b/IR_fused_for_CModel_params.npz'
QUAN:
    strategy: 'symmetry_max_t' # minmax, null, scale_shift, symmetry_max
    qconfig:
        weight_quan: 'perlayer' # perchannel , perlayer
        table: 'calibrations/resnet50_v1b.json'
        image_path: './models/image_for_calibrate'
EVALUATION:
    input_images: './models/image_for_compare'
    label_file: 'models/imagenet_lsvrc_2015_synsets.txt'
    image_file: 'models/imagenet_5000test.list'
    input_node: '0:0'
    output_node: '506:0'
SAVE_PATH: 'pbs/resnet50_v1b_quan.pb'
```
- MODEL
    - graph: moffett ir for cmodel graph file
    - params: moffett ir for cmodel params file
- QUAN
    - strategy: 量化策略
        - minmax: 很接近tensorflow的量化方案，我们不采用
        - null: 不做任何量化
        - scale_shift: 目前我们fpga采用的量化方案
        - symmetry_max: 我们芯片上预研的方案
        - {strategy}_t: 对应方案的true quantization方案，目前只支持symmetry_max
    - table: 校准产生的tabel文件 [calibration doc](http://192.168.1.51:3001/pD9MdylvSIWYL-OAis-44Q?view#How-VALUATIONto-use-calibration)
    - image_path: 分析量化误差所用的图片路径
- EVALUATION
    - input_node: graph node 0
    - output_node: graph node last
    - input_image: 分析最终网络误差
    - label_file: imagenet label file
    - image_file: imagenet image list file
- SAVE_PATH: quantization pb save path
 
qutization目前分为两个部分：
#### 面向backend

面向backend的部分：这部分主要逐层分析量化过程中每层带来的误差，输入为IR_fused_for_CModel，可参考[test_quantize_error.py](http://192.168.1.51:3001/6dNpM7L_Rdmuq7tYpYoWpg?both#test_quantize_errorpy)
```
python examples/test_quantize_error.py --config-file configs/resnet50_v1b.yml
```
这一步会逐层输出与没有做quantization的distance，最后会输出一个最终结果的distance
```
final distance 0.027732491493225098
```
distance 越小越好, 这一步会产生量产误差和最中量化的pb文件，将pb文件拷贝到MODEL_ZOO中去，类似如下
```
/hdd1/MODEL_ZOO/vision/classification/resnet50v1b/original/
├── moffett_ir
│   ├── IR_for_reconstruct_graph.json
│   ├── IR_for_reconstruct_graph.png
│   ├── IR_for_reconstruct_params.npz
│   ├── IR_fused_for_CModel_graph.json
│   ├── IR_fused_for_CModel_graph.png
│   └── IR_fused_for_CModel_params.npz
├── resnet50v1b-0000.params
├── resnet50v1b_symmetry_max_quantization.pb
└── resnet50v1b-symbol.json
```

#### 面向algorithm
面向algrithm的部分：这部分主要分析在目标task上带来的最终误差，输入为IR_fused_for_CModel，可参考[test_imagenet_quantize_acc.py](http://192.168.1.51:3001/6dNpM7L_Rdmuq7tYpYoWpg?both#test_imagenet_quantize_accpy)
```
python examples/test_imagenet_quantize_acc.py --config-file configs/resnet50_v1b.yml
```
这一步会产生我们目前所支持所有量化方案下测试集的accuracy,汇总这一步的结果参考[quantization results](http://192.168.1.51:3001/Po6E_ww5TEeAcInC_ylwiw)，同时补充table (scale_shift perlayer; symmertry_max perchannel)
| model          | compiler pass | torch rt | tf rt | quantacc (fpga/asic) |
| -------------- | ------------- | -------- | ----- | -------------------- |
| resnet50 v1b   | yes           | yes      | yes   | 0.765 / 0.761        |
| resnet50sp v1b | yes           | yes      | yes   | 0.681 / 0.732        |

### step7：modeling performance

### step8：fast prune(Experimental)
参考projects/fast_prune

