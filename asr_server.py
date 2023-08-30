"""
@Author: pkq1688
@Date: 2023-08-28 15:13:07
@LastEditors: pkq1688
@LastEditTime: 2023-08-28 19:17:31
@FilePath: /speech_clone/asr_server.py
@Description: 
"""
import os
import tempfile
import codecs
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.msdatasets import MsDataset

if __name__ == '__main__':
    param_dict = dict()
    # param_dict['hotword'] = "https://isv-data.oss-cn-hangzhou.aliyuncs.com/ics/MaaS/ASR/test_audio/hotword.txt"
    param_dict["hotword"] = "kw.txt"

    output_dir = "./output"
    batch_size = 1

    # dataset split ['test']
    ds_dict = MsDataset.load(dataset_name='speech_asr_aishell1_hotwords_testsets', namespace='speech_asr')
    work_dir = tempfile.TemporaryDirectory().name
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    # wav_file_path = os.path.join(work_dir, "wav.scp")
    wav_file_path = "data/2020-05-02 09_43_32.wav"


    # with codecs.open(wav_file_path, 'w') as fin:
    #     for line in ds_dict:
    #         wav = line["Audio:FILE"]
    #         idx = wav.split("/")[-1].split(".")[0]
    #         fin.writelines(idx + " " + wav + "\n")
    audio_in = wav_file_path

    inference_pipeline = pipeline(
        task=Tasks.auto_speech_recognition,
        model="damo/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
        output_dir=output_dir,
        batch_size=batch_size,
        param_dict=param_dict)

    rec_result = inference_pipeline(audio_in=audio_in)
    print(rec_result)