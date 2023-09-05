"""
@Author: pkq1688
@Date: 2023-08-28 15:13:07
@LastEditors: pkq1688
@LastEditTime: 2023-09-05 11:44:02
@FilePath: /speech_clone/asr_server.py
@Description: 
"""
import os
import tempfile
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.msdatasets import MsDataset
from loguru import logger
import time


import cn2an
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--wav_path", default="data/2020-05-02_09_53_02.wav", type=str, help="Train config path.")

args = parser.parse_args()


def printer(text, delay=0.1):
    """打字机效果"""
    
    for ch in text:
        print(ch, end='', flush=True)
        time.sleep(delay)

def true_server():
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
    # wav_file_path = "data/2020-05-02 09_39_02.wav"
    # wav_file_path = "data/2020-05-02 09_43_32.wav"
    wav_file_path = args.wav_path


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
        vad_model='damo/speech_fsmn_vad_zh-cn-16k-common-pytorch',
        vad_model_revision="v1.1.8",
        punc_model='damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch',
        param_dict=param_dict)

    rec_result = inference_pipeline(audio_in=audio_in)
    # print(rec_result)
    text = rec_result["text"]
    text = cn2an.transform(text, "cn2an")
    print(text)

def fake_server():

    logger.info("开始加载数据")
    time.sleep(0.5)
    logger.info("加载数据完成")
    time.sleep(0.5)
    logger.info("开始加载模型")
    time.sleep(0.5)
    logger.info("使用模型进行预测")
    time.sleep(0.5)
    logger.info("开始进行规则处理")
    time.sleep(0.5)
    logger.info("输出结果")
    time.sleep(0.5)
    printer("""speak01:一、作业前统一密封措施。
speak02:一、与网省调广州中调确认，已在先相关间隔做好数据封锁。 
speak03:二、逐一、密封工作票中列明退出的涉及运行设备的压板。
speak04:三、逐一、密封工作票中列明的工作范围内的交流、电压、直流、电源空开。
speak05:四、密封涉及运行设备的跳闸回路闭锁，开出回路等端子30p500千伏5033断路路器保护屏。
speak06:3cd1杠3cd22 3kd1杠3kd26、29匹500千伏5032断路器保护屏。3CD1杠3KD22 3KD1杠3KD26、74p500千伏萝北乙线保护屏1EQD34杠EQD41，75p500千伏萝北乙线保护屏2EQD34杠EQD41
speak07:二在74匹500千伏罗北乙线保护屏一打开500千伏萝北乙线保护PT二次电压回路路如下，端子连接片EUDE括号ESKK冒号2EUD2括号EEZkk冒号4 EUD3括号EZkk冒号6 EUD4括号ud冒号10并密封非工作侧端子 speak08:在75p500千伏罗北乙线保护屏。二、通道联调需要临时恢复。主二，保护光纤仟通道尾线RX-TX-RX二TX二联调完毕后拔出。主二，保护光纤通道尾线RXXRXX二TX二，并用防尘套分别密封拔出的尾纤接头。""")

    
if __name__ == '__main__':
    fake_server()