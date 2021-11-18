# Source-Aware Neural Speech Coding for Noisy Speech Compression
Yang, Haici, et al. "Source-Aware Neural Speech Coding for Noisy Speech Compression." ICASSP 2021-2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2021.
## Prerequisites
Python 3.6.8<br>
torch 1.6.0<br>
torchaudio 0.6.0

## Dataset 
- Speech: TIMIT([https://www.ldc.upenn.edu](https://www.ldc.upenn.edu))<br>
- Noise: Duan stational noise ([http://www2.ece.rochester.edu/~zduan/is2012/examples.html](http://www2.ece.rochester.edu/~zduan/is2012/examples.html))
  - <em>Duan, Zhiyao, Gautham J. Mysore, and Paris Smaragdis. "Speech enhancement by online non-negative spectrogram decomposition in nonstationary noise environments." In Thirteenth Annual Conference of the International Speech Communication Association. 2012.</em>
## Model training
Main hyper-parameters and their default setting for model training:
| Symbol | Description |
| --- | ----------- |
| filters = 100           |  Output channel size of encoder|
| d = 1                   |  Dimension of the codec|
| m = 32                  |  The number of codes in the code book|
| sr = True               |  To do super-resolution based downsampling or not|
| lr = 0.0001             |  Learning rate |
| br = 8                  |  Bitrate(khz) |
| scale = 1000            |  Scale to control the hardness of the softmax function. |
| label = time.strftime("%m%d_%H%M%S") |  Model label|
| weight_mse = 30         |  Loss weight for MSE(waveforms) term|
| weight_mel = 0.5        |  Loss weight for mel-spectogram term|
| weight_qtz = 0.5        |  Loss weight for quantization|
| weight_etp_total = 0.1  | Loss weight for the total entropy|
| weight_etp_ratio = 0.05 | Loss weight for the entropy ratio between source and noise|
| ratio = 1.0             | Ratio of assigned bitrate between source and noise|
| update_ratio = False    | Whether update the ratio during training or not|
| db = 0                  | Initial SDR of input data, 0 or 5|

Train proposed model, <code>python3 train_model.py</code>.<br>
Train baseline model, <code>python3 train_base.py</code>.

