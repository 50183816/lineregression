from pydub import AudioSegment
import io
from python_speech_features import mfcc
import numpy as np

song = AudioSegment.from_file('../datas/music/我们的纪念.wav')
framerate = song.frame_rate
width = song.sample_width
channel = song.channels
duration = len(song)
print('frame_rate:{}, sample_width:{},channel:{},durantion:{}'.format(framerate,width,channel,duration))

data = np.array(song.get_array_of_samples()).reshape(-1,channel)
print(data.shape)
"""
signal: 给定音频文件的数据，是一个数组的形式
samplerate: 给定音频文件的频率值
numcep：默认值13，给定每帧数据划分为多少个区间，相当于考虑13个不同的频率(Hz)区间上的振幅值
nfft: 是在做傅里叶变换过程中的参数值
winstep: 给定划分过程中，区间的步长。也就是隔多久划分一个time区间考虑(相当于作为一帧考虑)，给定的单位是秒，默认是0.01秒
winlen: 给定每个帧的时间长度，单位秒，默认为0.025秒
"""
mfcc_feature = mfcc(signal=data,samplerate=framerate,numcep=13,nfft=2048,winlen=0.025,winstep=0.01)
print(mfcc_feature.shape)