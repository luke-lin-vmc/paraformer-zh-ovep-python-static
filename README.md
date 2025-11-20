# About paraformer-zh-ovep-python-static
This Python pipeline is to show how to run paraformer on Intel CPU/GPU/NPU thru [ONNX Runtime](https://github.com/microsoft/onnxruntime) + [OpenVINO Execution Provider](https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html)

This implementation is forked from [RKNN implementation](https://github.com/k2-fsa/sherpa-onnx/tree/bb96ea34bbf7f97ffd076bee69814b4c68b67558/scripts/paraformer/rknn) of [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) project 

### Key features
* Support Chinese (zh) only
* Models are converted to static (mainly for NPU)

# Quick Steps
## Download models
Visit https://huggingface.co/funasr/paraformer-zh/tree/main, download the followning 
files
```
am.mvn
config.yaml
configuration.json
model.pt
seg_dict
tokens.json
```
The project directory should look like
```
C:\Github\paraformer-zh-ovep-python-static>dir
 Volume in drive C is InstallTo
 Volume Serial Number is 76DF-BB22

 Directory of C:\Github\paraformer-zh-ovep-python-static

11/20/2025  02:31 PM    <DIR>          .
11/20/2025  02:27 PM    <DIR>          ..
11/20/2025  02:28 PM            11,203 am.mvn
11/18/2025  04:31 PM           417,742 asr_example.wav
11/20/2025  02:29 PM             2,509 config.yaml
11/20/2025  02:29 PM               472 configuration.json
11/14/2025  02:30 PM             1,169 export_decoder_onnx.py
11/18/2025  01:55 PM             5,189 export_encoder_onnx.py
11/14/2025  02:30 PM             1,510 export_predictor_onnx.py
11/20/2025  02:31 PM       880,502,012 model.pt
11/20/2025  02:24 PM             4,682 README.md
11/20/2025  01:48 PM               128 requirements.txt
11/20/2025  02:30 PM         8,287,834 seg_dict
11/19/2025  05:32 PM            10,969 test_onnx.py
11/20/2025  02:31 PM            93,676 tokens.json
11/18/2025  02:19 PM            46,002 torch_model.py
              14 File(s)    889,385,097 bytes
               2 Dir(s)  232,529,977,344 bytes free
```
## Export models
Run the following commands to export models
```
pip install -r requirements.txt
python export_encoder_onnx.py --input-len-in-seconds 5
python export_decoder_onnx.py --input-len-in-seconds 5
python export_predictor_onnx.py --input-len-in-seconds 5
```
* As NPU does not support dynamic input shape, we need to convert it to static by specifying a fixed input length. You man change the length by using  "```--input-len-in-seconds <length>```" per your requirement.

Models (```*.onnx```) will be exported under the same directory
```
encoder-5-seconds.onnx
decoder-5-seconds.onnx
predictor-5-seconds.onnx
```
## Run
Usage
```
Usage: python test_onnx.py --device <DEVICE> --input-len-in-seconds <LENGTH> <sound_file>
```
* Supported devices: ```CPU```, ```GPU``` and ```NPU```. If ```--device``` is not specified, CPUExecutionProvider will be used by default<br> 

Run on CPU
```
python test_onnx.py --device CPU --input-len-in-seconds 5 asr_example.wav
```
Run on GPU
```
python test_onnx.py --device GPU --input-len-in-seconds 5 asr_example.wav
```
Run on NPU
```
python test_onnx.py --device NPU --input-len-in-seconds 5 asr_example.wav
```
:warning:[NOTE] The 1st time running on NPU will take long time (about 3 minutes) on model compiling. [OpenVINO Model Caching](https://docs.openvino.ai/2025/openvino-workflow/running-inference/optimize-inference/optimizing-latency/model-caching-overview.html) has been enabled for NPU to ease the issue. This feature will cache compiled models. Although the 1st run still takes long, but later runs can be faster as model compilation has been skipped.
## Tested devices
The test was done on a ```Intel(R) Core(TM) Ultra 7 268V (Lunar Lake)``` system, with
* ```iGPU: Intel(R) Arc(TM) 140V GPU, driver 32.0.101.8247 (10/22/2025)```
* ```NPU: Intel(R) AI Boost, driver 32.0.100.4404 (11/7/2025)```
### Result
| Model      | CPU  | GPU | NPU |
|------------|------|-----|-----|
| Paraformer | OK   | OK  | OK  |

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
features.shape (83, 560)
[[10.330146  11.28574   11.197521  ... 11.283754  11.682403  11.068745 ]
 [ 8.023468   9.23599   11.084137  ... 11.941153  12.421517  11.915011 ]
 [ 5.4080577  7.2164407  9.937752  ... 13.703872  12.714883  12.267856 ]
 ...
 [14.420621  16.180243  19.120308  ... 12.683482  11.631289  11.296776 ]
 [11.864075  13.517877  16.202063  ... 11.239302  11.938505  11.994032 ]
 [ 8.41748   10.243659  12.98756   ... 11.673848  11.6645775 11.501234 ]]
encoder_out.shape (1, 83, 512)
encoder_out.sum 64.594604 0.0015200161
[[[-0.01406524  0.00265226 -0.00964681 ...  0.01442128 -0.03431666
    0.00986198]
  [-0.03563054  0.04568488  0.01158986 ... -0.05193643 -0.06663334
    0.02791104]
  [-0.04298359  0.06046052  0.00973543 ... -0.05894379 -0.06061184
    0.01853588]
  ...
  [-0.00860532 -0.00040867 -0.03937218 ...  0.02093086 -0.01048481
   -0.02408925]
  [-0.01257268 -0.00773508 -0.03052646 ...  0.02829604  0.00282896
    0.00458103]
  [-0.03672801  0.05195791 -0.04131586 ... -0.00337254  0.04668033
    0.07417187]]]
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
[Full log](https://github.com/luke-lin-vmc/whisper-ovep-python-static/blob/main/log_full.txt) (from scratch) is provided for reference

## Known issues
1. The following warning appears when running the pipeline thru OVEP for the 1st time
```
C:\Users\...\site-packages\onnxruntime\capi\onnxruntime_inference_collection.py:123:
User Warning: Specified provider 'OpenVINOExecutionProvider' is not in available provider names.
Available providers: 'AzureExecutionProvider, CPUExecutionProvider'
```
Solution is to simply reinstall ```onnxruntime-openvino```
```
pip uninstall -y onnxruntime-openvino
pip install onnxruntime-openvino
```
2. Only Arc iGPUs (Meteor Lake, Lunar Lake, Panther Lake and Arrow Lake H-series) are supported. Running on unsupported iGPU (such like Iris Xe or UHD) may lead to incorrect output, such as "!!!!!!!!!!!!!!".

