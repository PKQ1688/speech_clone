import os
from modelscope.tools import run_auto_label
from loguru import logger
import numpy as np

from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.utils.audio.audio_utils import TtsTrainType

from modelscope.models.audio.tts import SambertHifigan
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks

from scipy.io.wavfile import write


def get_label_data(input_wav, output_data):
    """获取标注数据"""
    # 标注数据3
    if not os.path.exists(output_data):
        # os.system("rm -rf {}".format(output_data))
        os.mkdir(output_data)
        ret, report = run_auto_label(input_wav=input_wav, work_dir=output_data, resource_revision="v1.0.6")
        logger.info("标注数据保存路径: {}".format(output_data))
    else:
        logger.info("标注数据已存在: {}".format(output_data))


def train_model(output_data, user_name):
    pretrained_model_id = 'damo/speech_personal_sambert-hifigan_nsf_tts_zh-cn_pretrain_16k'

    dataset_id = output_data
    pretrain_work_dir = user_name

    if os.path.exists(pretrain_work_dir):
        os.system("rm -rf {}".format(pretrain_work_dir))

    os.mkdir(pretrain_work_dir)

    # 训练信息，用于指定需要训练哪个或哪些模型，这里展示AM和Vocoder模型皆进行训练
    # 目前支持训练：TtsTrainType.TRAIN_TYPE_SAMBERT, TtsTrainType.TRAIN_TYPE_VOC
    # 训练SAMBERT会以模型最新step作为基础进行finetune
    train_info = {
        TtsTrainType.TRAIN_TYPE_SAMBERT: {  # 配置训练AM（sambert）模型
            'train_steps': 202,  # 训练多少个step
            'save_interval_steps': 200,  # 每训练多少个step保存一次checkpoint
            'log_interval': 10  # 每训练多少个step打印一次训练日志
        }
    }

    # 配置训练参数，指定数据集，临时工作目录和train_info
    kwargs = dict(
        model=pretrained_model_id,  # 指定要finetune的模型
        model_revision="v1.0.6",
        work_dir=pretrain_work_dir,  # 指定临时工作目录
        train_dataset=dataset_id,  # 指定数据集id
        train_type=train_info  # 指定要训练类型及参数
    )

    trainer = build_trainer(Trainers.speech_kantts_trainer,
                            default_args=kwargs)

    trainer.train()


def model_infer(model_dir, content="今天的天气真不错", output_wav="output.wav"):
    model_dir = os.path.abspath(model_dir)

    custom_infer_abs = {
        'voice_name':
            'F7',
        'am_ckpt':
            os.path.join(model_dir, 'tmp_am', 'ckpt'),
        'am_config':
            os.path.join(model_dir, 'tmp_am', 'config.yaml'),
        'voc_ckpt':
            os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan', 'ckpt'),
        'voc_config':
            os.path.join(model_dir, 'orig_model', 'basemodel_16k', 'hifigan',
                         'config.yaml'),
        'audio_config':
            os.path.join(model_dir, 'data', 'audio_config.yaml'),
        'se_file':
            os.path.join(model_dir, 'data', 'se', 'se.npy')
    }
    kwargs = {'custom_ckpt': custom_infer_abs}

    model_id = SambertHifigan(os.path.join(model_dir, "orig_model"), **kwargs)

    inference = pipeline(task=Tasks.text_to_speech, model=model_id)
    output = inference(input=content)
    output_data = np.frombuffer(output["output_wav"], dtype=np.int16)
    write(output_wav, 16000, output_data)

    # return output_data.tolist()


if __name__ == "__main__":
    input_wav = "data/speech_clone/diandian"
    output_data = "data/speech_clone/diandian_label"
    # get_label_data(input_wav, output_data)
    usr = "model/speech_clone/diandian"
    # train_model(output_data=output_data, user_name=usr)
    model_infer(model_dir=usr,content="8月10日，《每日经济新闻》记者来到湖北黄石，实地探访了处于漩涡中的黄石市鸿泰公共巴士有限公司（以下简称黄石鸿泰）。记者在该公司的停车场看到，烈日之下，数十辆停止运营的公交车旁边荒草丛生，车内积满了灰尘。",output_wav="output.wav")
