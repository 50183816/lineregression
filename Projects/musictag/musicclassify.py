# _*_ codig utf8 _*_
import numpy as np
from pydub import AudioSegment
from python_speech_features import mfcc
import warnings
import pandas as pd
from glob import glob
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import  Pipeline
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
# np.mean()
# np.diag()
music_info_file_path='../datas/music/music_info.csv'
music_file_dir = 'F:\\BaiduNetdiskDownload\\项目二：音乐文件分类\\05_随堂代码\\music\\data\\music\\*.mp3'

# load music info
musicInfo = pd.read_csv(music_info_file_path)
# print(musicInfo)
lblEncoder = LabelEncoder()
print(musicInfo.iloc[:,-1])
result = lblEncoder.fit_transform(musicInfo.iloc[:,-1])
print(result)
musicInfo['tag_value'] = result
#处理音乐文件
files = glob(music_file_dir)
results=[]
classes=[]
for file in files:
    try:
        music = AudioSegment.from_mp3(file)
        music = music.set_channels(2)
        music = music.set_frame_rate(44100)
        music = music.set_sample_width(2)
        filename = os.path.split(file)[-1]
        filename = filename.split('-')[-1].replace('.mp3','').strip()
        tag = musicInfo[musicInfo.name==filename].reset_index()['tag_value']
        if len(tag)>0:
            tag = tag[0]
        else:
            tag = -1
        classes.append(tag)
        # print(tag)
        # print('{}: {}'.format(filename,tag))
        wav_file_path = '../datas/music/{}.wav'.format(filename)
        outf = music.export(wav_file_path,format='wav',tags=['转换格式'])
        music = AudioSegment.from_wav(wav_file_path)
        print('wav file duration:{} s'.format(music.duration_seconds))
        data = np.array(music.get_array_of_samples()).reshape(-1, music.channels)
        print(data.shape)
        features = mfcc(signal=data,samplerate=music.frame_rate,numcep=26,winstep=1,nfft=2048)
        print(features.shape)
        trans_features = np.transpose(features)
        result = np.mean(trans_features,axis=1)
        result_cov = np.cov(trans_features)
        for i in np.arange(result_cov.shape[0]):
            result = np.append(result,np.diag(result_cov,k=i))

        # print(result.shape)
        results.append(result)
    except Exception as ex:
        print('convert mp3 to wav throws exception {}: message:{}'.format(file,ex))
print(np.shape(results))
np.save('../datas/music/sample_features',results)
print(classes)
np.save('../datas/music/sample_targets', classes)
np.save('../datas/music/music_features_meta',musicInfo)
    # features = mfcc(signal=data,samplerate=music.frame_rate,numcep=26,nfft=2048)
    # print(features.shape)

pipline = Pipeline(steps=[
    ('pca',PCA(n_components=20,random_state=22,whiten=False)),
    ("svc",SVC(C=1,kernel='rbf',degree=3,gamma='auto_deprecated',decision_function_shape='ovr',random_state=22))])
params = {
    'pca__n_components':[0.3,0.5],
    'svc__C':[1,0.5,0.05],
    'svc__kernel':['rbf','linear'],
    'svc__degree':[3,4],
    'svc__gamma':['auto',0.1,0.3],
}
gridSearch = GridSearchCV(estimator=pipline,param_grid=params)
gridSearch.fit(results,classes)
print(gridSearch.best_params_)
print('done!!!')

