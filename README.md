[TOC]

This repo provides various tools for neural network front-end compiler, quantization and framework converter among tensorflow, pytorch and mxnet

An example of a resnet50 model from mxnet is used to demo the above mentioned tools 

### Resnet50 checklist
## Get Environment Docker 
### Get Docker Image On server 192.168.1.143

使用前执行```groups```，查看自己是否在docker用户组中，如果不在，可向相关人申请权限，然后执行```docker run -ti -v /path/to/your/file/directory:/directory_name moffett/frontendcompiler:0.0.2 /bin/bash```创建容器

### Pull Image from Dockerhub

拉取image```docker pull xhxian/frontendcompiler:0.0.1```，然后执行```docker run -ti -v /path/to/your/file/directory:/directory_name xhxian/frontendcompiler:0.0.1 /bin/bash```创建容器

### Use Dockerfile

There is a DockerFile in GitLab SoftwareTeam/FrontendCompile
-  To build docker image
```docker build --tag moffett/frontendcompiler:0.0.1```
-  To bring up a docker container for development, mount your file directory to docker
```
docker run -ti -v /your/file/directory:/directory_name \ 
        moffett/frontendcompiler:0.0.1 /bin/bash
```

## Get Frontend Compiler

创建容器并启动容器
```docker run -ti -v -name xxxxx /your/file/directory:/directory_name moffett/frontendcompiler:0.0.1 /bin/bash```

退出容器 ```exit```

克隆frontendcompiler tool到容器中去
```
cd /your/file/directory
user in shenzhen
    git clone git@192.168.1.51:lizhilong/frontendcompiler.git -b quantization_and_reconstruct
user not in shenzhen 
    git clone git@sz.server.techx.cn:lizhilong/frontendcompiler.git -b quantization_and_reconstruct
```
再进入容器就可以看到```/directory_name/frontendcompiler```

## Frontend compiler usage

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
- configs
包含一些搭配工具使用的配置文件
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
这些example详细文档可参考[Frontend Compiler Example Documentation](http://192.168.1.51:3001/6dNpM7L_Rdmuq7tYpYoWpg)

- projects
```
|-- fast_prune
|   `-- fast_pruning
|       |-- core
|       |-- data
|       |-- optimizer
|       `-- pruning
|-- tensorflow
|   `-- cifar10
|       `-- include
|-- torch
    `-- cifar10
```
目前project里面包含了三个demo: fast prune on resnet50v1b, tensorflow reconstruct, pytorch reconstruct

- src
```
|-- compiler
|-- dataset
|-- np_ops
|-- postprocess
|   `-- tf
|-- quantization
|-- reconstructor
|-- utils
```
src实现了frontend compiler的核心功能，其中主要包含了compiler，quantization，reconstructor，utils等模块。详细文档参考[Frontedn Compiler Core Documentation](http://192.168.1.51:3001/pD9MdylvSIWYL-OAis-44Q)
- tools (废弃)

## CheckList

![](/uploads/upload_e7912f8523200a5a6bf95b1b0a6bfef3.png)

以下将以gluoncv resnet50v1b为例产生如下的checklist
### step1: get original model
在gluoncv中下载原始模型，得到一个accuracy = 77.67的model

### step2: sparse training
- Prune Interface
![](/uploads/upload_fd7d173c02dd71b1713fe6c61df1cb96.png)
- Generate Sparse Dict
![](/uploads/upload_f0b694f9c63eb4a64354741293dea3d5.png)
- Prune usecase
![](/uploads/upload_768672b48da6d0f0a05e57dedfe7c5d9.png)

- training
    - 创建Prune实例的主要参数，目前只需关注model，pretrain_step，sparse_step，frequency，prune_dict：
	    - model为需要进行压缩的模型
	    - pretrain_step为训练开始后不需要执行压缩的训练轮数
	    - sparse_step为经过pretrain_step指定轮数后开始执行压缩的轮数
	    - frequency与sparse_step配合，每训练frequency轮后压缩一次
	    - prune_dict是需要进行压缩的参数和相应压缩率构成的字典
    - Prune实例的使用方法
        - 定义压缩规则，图中所示的规则为只压缩卷积层，压缩率为且第一层卷积层不进行压缩，具体规则定义需要分析各层参数的名称和性质
        - 创建Prune实例，实例应在正式开始循环训练前创建
        - 完成一个batch数据的训练并进行损失值反向传播后调用Prune实例的prepare方法
        - trainer进行step()后再进行prune操作
    - Run 
    
- Visualize model
    - 保存训练好的模型：
	    - 模型训练完成后使用net.save_parameters()并指定文件名来保存训练好的模型参数，后	缀为*.params
	- 使用gluoncv来转化保存好的模型参数文件
	    - gluoncv.utils.export_block(save_name, net, data_shape, epoch, preprocess, layout)导出mxnet支持的symbol graph
	- 使用Netron观察压缩后的模型参数
	    - 将上一步骤获得的*.params文件导入Netron中，并观察各层参数状况来判断压缩是否成功

- evalution
重复多次training过程将得到不同稀疏率的压缩模型

| model        | sparse rate | acc top1 | training date | input_shape |
| ------------ | ----------- | -------- | ------------- | ----------- |
| resnet50 v1b | 0           | 77.67    | xx            | 224x224     | 
| resnet50 v1b | 93.75       | 74.12    | 20200712      | 224x224     |
| resnet50 v1b | 90          | xxxxx    | 20200712      | 224x224     |

每次得到效果理想可以发布的模型，可应该提交到MODEL_ZOO
MODEL_ZOO的目录结构如下：
```
├── nlp
├── recommendation
└── vision
```
resnet50正常提交产生的目录结构应该如下：
```
/hdd1/MODEL_ZOO/vision/classification/resnet50v1b/
├── 20200717_sp93.75_224x224
│   ├── resnet50v1b_sparse_93.75-0199.params
│   └── resnet50v1b_sparse_93.75-symbol.json
└── original
    ├── resnet50v1b-0000.params
    └── resnet50v1b-symbol.json
```
模型目录的结构命名{training date}_sp{sparse rate}_{input_shape}

### step3: frontend model compiler test

使用frontend compiler examples中的test_compiler脚本
```python ./examples/test_complier.py mxnet resnet50v1b 0 ```
会在当前目录下生成moffett_ir目录，目录结构如下：
```
├── IR_for_reconstruct_graph.json
├── IR_for_reconstruct_graph.png
├── IR_for_reconstruct_params.npz
├── IR_fused_for_CModel_graph.json
├── IR_fused_for_CModel_graph.png
└── IR_fused_for_CModel_params.npz
```
如果没有产生上述结果，则反馈至@小鹤,反之则补足表格
| model        | acc top1 | orig model top1 | training date | input_shape | compiler pass |
| ------------ | -------- | --------------- | ------------- | ----------- | ------------- |
| resnet50 v1b | 74.12    | 77.67           | 20200712      | 224x224     | yes           |
| resnet50 v1b | xxxxx    | 77.67           | 20200712      | 224x224     | yes           |

### step4: model complexity
此部分将会统计该模型的nnz, sparsity, dense flops, sparse flops
```
python examples/test_model_complexity.py 
        --graph-json moffett_ir/IR_for_reconstruct_graph.json
        --params-file moffett_ir/IR_for_reconstruct_params.npz
```
nnz: 3522685, sparsity: 0.8618712639560533, dense_flops: 3948251920.0, sparse_flops: 357813835.0
补充table:
| model        | training date | input_shape | compiler pass | nnz/sparsity   | flops dense/sp M |
| ------------ | ------------- | ----------- | ------------- | -------------- | ---------------- |
| resnet50 v1b | 20200712      | 224x224     | yes           | TBB            | TBB              |
| resnet50 v1b | 20200712      | 224x224     | yes           | 3522685/0.8618 | 3948/358         | 

### step5: reconstructor test 
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

