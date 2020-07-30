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

example codes for using the compiler, quantization and reconstruction tools

More details can be found in [Frontend Compiler Example Documentation](examples/Frontend_Compiler_Examples.md)
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

### step 1: get original model

we use gluoncv resnet50v1b (top1=77.67 on imagenet) as an example here

### step 2: prune the dense network using the [sparse tools](https://github.com/jiacliu09/PruningTools) provided by Moffett AI 

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

### step 3: frontend model compiler test
compile the sparse mxnet model into moffett IR

```python ./examples/test_compiler.py mxnet resnet50v1b 0 ```

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
Convert Moffett IR to pytorch, tensorflow models for training and inference
```
In pytorch:
python examples/test_pytorch_reconstructor.py 
        --graph moffett_ir/IR_for_reconstruct_graph.json 
        --params moffett_ir/IR_for_reconstruct_params.npz 
        --input_npy input.npy 
        --result_npy result.npy
```

```
In tensorflow:
python examples/test_tensorflow_reconstructor.py 
        --graph moffett_ir/IR_for_reconstruct_graph.json 
        --params moffett_ir/IR_for_reconstruct_params.npz 
        --input_npy input.npy 
        --result_npy result.npy 
        --save_path tf_reconstruct.pb
```

### step 6: Quantization test
Layer-wise comparison of feature maps before and after quantization. The smaller the cosine distance, the better quantization is achieved. 

```
python examples/test_quantize_error.py --config-file configs/resnet50_v1b.yml
```
The expected output after quantization test:
```
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

An example of the quantization configure file
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
    - strategy: 
        - minmax: 
        - null: 
        - scale_shift: 
        - symmetry_max: 
        - {strategy}_t: 
    - table: table to save calibration results 
    - image_path: image path for calibrition 
- EVALUATION
    - input_node: graph node 0
    - output_node: graph node last
    - input_image: evaluation images 
    - label_file: imagenet label file
    - image_file: imagenet image list file
- SAVE_PATH: quantization pb save path
 
