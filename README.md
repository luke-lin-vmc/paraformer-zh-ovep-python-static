# About paraformer-zh-ovep-python-static
This Python pipeline is to show how to run Paraformer ASR on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from [RKNN implementation](https://github.com/k2-fsa/sherpa-onnx/tree/bb96ea34bbf7f97ffd076bee69814b4c68b67558/scripts/paraformer/rknn) of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project 

Audio samples ("```0.wav```, ```1.wav``` and ```2.wav```") are downloaded from https://huggingface.co/csukuangfj/paraformer-onnxruntime-python-example/tree/main

## Key features
* Off-line (non-streaming) mode
* Support Chinese (zh) only
* Models are converted to static (mainly for NPU)

# Quick Steps
## Download and export models
Visit https://huggingface.co/funasr/paraformer-zh/tree/main, download the following 
files
```
am.mvn
config.yaml
configuration.json
model.pt
seg_dict
tokens.json
```
Run the following commands to export models
```
pip install -r requirements.txt
python export_encoder_onnx.py --input-len-in-seconds 5
python export_decoder_onnx.py --input-len-in-seconds 5
python export_predictor_onnx.py --input-len-in-seconds 5
```
* As NPU does not support dynamic input shape, it is required to convert to static by specifying a fixed input length. You may configure the input length by setting  "```--input-len-in-seconds <length>```" per your requirement.

The following models (```*.onnx```) will be exported under the same directory
```
encoder-5-seconds.onnx
decoder-5-seconds.onnx
predictor-5-seconds.onnx
```
The project directory should look like
```
(openvino_venv) C:\Github\paraformer-zh-ovep-python-static>dir
 Volume in drive C is InstallTo
 Volume Serial Number is 76DF-BB22

 Directory of C:\Github\paraformer-zh-ovep-python-static

11/21/2025  11:51 AM    <DIR>          .
11/20/2025  04:20 PM    <DIR>          ..
11/21/2025  11:32 AM           179,712 0.wav
11/21/2025  11:32 AM           165,042 1.wav
11/21/2025  11:32 AM           144,922 2.wav
11/21/2025  11:44 AM            11,203 am.mvn
11/21/2025  11:44 AM             2,509 config.yaml
11/21/2025  11:44 AM               472 configuration.json
11/21/2025  11:50 AM       228,460,151 decoder-5-seconds.onnx
11/21/2025  11:50 AM       632,885,122 encoder-5-seconds.onnx
11/21/2025  11:32 AM             1,169 export_decoder_onnx.py
11/21/2025  11:32 AM             5,189 export_encoder_onnx.py
11/21/2025  11:32 AM             1,510 export_predictor_onnx.py
11/21/2025  11:45 AM       880,502,012 model.pt
11/21/2025  11:51 AM         3,152,772 predictor-5-seconds.onnx
11/21/2025  11:32 AM             6,446 README.md
11/21/2025  11:32 AM               120 requirements.txt
11/21/2025  11:45 AM         8,287,834 seg_dict
11/21/2025  11:32 AM            10,969 test_onnx.py
11/21/2025  11:45 AM            93,676 tokens.json
11/21/2025  11:32 AM            44,688 torch_model.py
11/21/2025  11:50 AM    <DIR>          __pycache__
              19 File(s)  1,753,955,518 bytes
               3 Dir(s)  225,971,204,096 bytes free
```
## Run
Usage
```
Usage: python test_onnx.py --device <device> --input-len-in-seconds <length> <sound_file>
```
* Supported devices: ```CPU```, ```GPU``` and ```NPU```. If ```--device``` is not specified, the default CPUExecutionProvider will be used 
* The ```--input-len-in-seconds``` value must match the value used when exporting the models

Run on CPU
```
python test_onnx.py --device CPU --input-len-in-seconds 5 1.wav
```
Run on GPU
```
python test_onnx.py --device GPU --input-len-in-seconds 5 1.wav
```
Run on NPU
```
python test_onnx.py --device NPU --input-len-in-seconds 5 1.wav
```
:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation has been skipped.
## Tested devices
The pipeline has been verified working on a ```Intel(R) Core(TM) Ultra 7 268V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 140V GPU, driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4404 (11/7/2025)```
### Result
| Sample | CPU | GPU | NPU |
|--------|-----|-----|-----|
| 0.wav  | OK  | OK  | OK  |
| 1.wav  | OK  | OK  | OK  |
| 2.wav  | OK  | OK  | OK  |

### Sample log (device is NPU)
```
(openvino_venv) C:\Github\paraformer-zh-ovep-python-static>python test_onnx.py --device NPU --input-len-in-seconds 5 1.wav
num_frames 500
num_input_frames 83
features sum 612931.0 (516, 80)
here (85, 560) True
features.shape (83, 560) (83, 560)
sum 700035.94 15.061014 17006.455 0.36588758
Device: OpenVINO EP with device = NPU
init encoder: ./encoder-5-seconds.onnx
init decoder: ./decoder-5-seconds.onnx
init predictor: ./predictor-5-seconds.onnx
---encoder---
NodeArg(name='x', type='tensor(float)', shape=[1, 83, 560])
-----
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 83, 512])
---decoder---
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 83, 512])
NodeArg(name='acoustic_embedding', type='tensor(float)', shape=[1, 83, 512])
NodeArg(name='mask', type='tensor(float)', shape=[83])
-----
NodeArg(name='decoder_out', type='tensor(float)', shape=[1, 83, 8404])
---predictor---
NodeArg(name='encoder_out', type='tensor(float)', shape=[1, 83, 512])
-----
NodeArg(name='alphas', type='tensor(float)', shape=[1, 83])
encoder_out.shape (1, 83, 512)
encoder_out.sum 64.594604 0.0015200161
alpha.shape (1, 83)
alpha.sum() 26.999882 0.32529977
acoustic_embedding.shape (26, 512)
padding.shape (57, 512) (26, 512)
acoustic_embedding.shape (83, 512)
acoustic_embedding.sum 17.41497 0.00040980257
[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.
 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.
 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
decoder_out encoder_out acoustic_embedding
decoder_out (1, 83, 8404)
decoder_out.sum 459326.0 0.6585017
[3055, 278, 7404, 6801, 3806, 1680, 6184, 4390, 1409, 7827, 7118, 7404, 991, 6426, 4803, 1074, 2773, 3595, 8069, 2483, 1320, 1980, 6857, 4500, 1383, 2] --> 26
['重', '点', '呢', '想', '谈', '三', '个', '问', '题', '首', '先', '呢', '就', '是', '这', '一', '轮', '全', '球', '金', '融', '动', '荡', '表', '现']
重点呢想谈三个问题首先呢就是这一轮全球金融动荡表现
```
[Full log](https://github.com/luke-lin-vmc/paraformer-zh-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Known issues
If the following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
This would be caused by that both ```onnxruntime``` and ```onnxruntime-openvino``` are installed. Solution is to remove both of them then re-install ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
Or simply to re-install ```onnxruntime-openvino``` if you would like to keep ```onnxruntime```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino~=1.23.0
```
