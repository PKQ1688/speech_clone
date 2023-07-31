import os
import gradio as gr
import numpy as np
from use_model_api import get_data_infer,get_data_label,get_data_train
from loguru import logger

from audio_utils import ndarray_to_wav

# import ssl
# model_list = os.listdir("model/speech_clone")
# logger.info(model_list)


def upload_records(record, index, cur_workspace: str):
    if record is None:
        return 0
    print("record: {}".format(record), flush=True)
    sr, data = record
    #  wavfile.write(osp.join(cur_workspace, "raw_" + str(index) + '.wav'), sr, data)

    # two channels
    if len(data.shape) > 1:
        single_data = data[:, 0]
    else:
        single_data = data

    target_sr = 16000
    wav = ndarray_to_wav(single_data, sr, target_sr)
    file = os.path.join(cur_workspace, str(index) + ".wav")
    with open(file, "wb") as f:
        f.write(wav)
    return 1


def launch_training_task(use_name, *audio_lst):
    valid_wav_entries = 0
    wav_data_path = "data/speech_clone/" + use_name

    if os.path.exists(wav_data_path):
        os.system("rm -rf {}".format(wav_data_path))
    os.mkdir(wav_data_path)

    logger.info(wav_data_path)

    for num in range(len(audio_lst)):
        res = upload_records(audio_lst[num], num, wav_data_path)
        valid_wav_entries += res

    # if valid_wav_entries < 10:
        # return "Error: 录音文件数量不足，请至少录制10个以上的音频文件"
    # yield "开始标注数据"
    train_progress = "开始标注数据"
    logger.info(train_progress)

    wav_data_path_label = wav_data_path + "_label"
    get_data_label(input_wav=wav_data_path, output_data=wav_data_path_label)

    model_path = "model/speech_clone/" + use_name
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    get_data_train(output_data=wav_data_path_label, model_name=model_path)


def launch_infer_task(model_name, content):
    logger.info(model_name)
    output_wav_path = "result/" + model_name[0]
    logger.info(output_wav_path)
    if not os.path.exists(output_wav_path):
        os.mkdir(output_wav_path)

    logger.info("开始推理")
    logger.info("模型路径:{}".format(model_name))
    logger.info("推理文本:{}".format(content))
    logger.info("输出音频路径:{}".format(output_wav_path))

    get_data_infer(
        model_name="model/speech_clone/"+model_name[0], content=content, output_wav_path=output_wav_path+"/output.wav"
    )
    # wav_res = np.array(output_wav)

    return output_wav_path+"/output.wav"


with gr.Blocks(css="#warning {color: red} .feedback {font-size: 24px}") as demo:
    gr.Markdown("## 灵动数上声音克隆")
    voice: str = ""
    # model_list = PRETRAINED_MODEL_ID
    my_models = []
    session = gr.State(
        {
            "modelscope_uuid": "guest",
            "modelscope_request_id": "test",
            "voice": voice,
            # 'model_list': model_list,
            "my_models": my_models,
        }
    )

    with gr.Tabs():
        with gr.TabItem("\N{rocket}模型定制") as tab_train:
            helper = gr.Markdown(
                """
            \N{glowing star}**定制步骤**\N{glowing star}

            Step 1. 录制音频\N{microphone}，点击下方音频录制并朗读左上角文字, 请至少录制10句话

            Step 2. 输入声音的名字作为唯一标识

            Step 3. 点击 **[开始训练]** \N{hourglass with flowing sand}，启动模型训练，等待约20分钟

            Step 4. 切换至 **[模型体验]** \N{speaker with three sound waves}，选择训练好的模型，感受效果

            \N{electric light bulb}**友情提示**\N{electric light bulb}

            \N{speech balloon}  朗读时请保持语速、情感一致

            \N{speaker with cancellation stroke}  尽量保持周围环境安静，避免噪音干扰

            \N{headphone}  建议佩戴耳机，以获得更好的录制效果
            """
            )
            with gr.Row():
                with gr.Column(scale=1):
                    audio_lst1 = [
                        gr.Audio(source="microphone", label="1. 希望我们大家都能像他一样"),
                        gr.Audio(
                            source="microphone", label="2. 不行, 他想了一下, 我不能这样对国王说, 这是在撒谎"
                        ),
                        gr.Audio(source="microphone", label="3. 但他们非常和气地问她说, 你叫什么名字"),
                        gr.Audio(source="microphone", label="4. 鸭子心想, 我必须去拿回我的软糖豆"),
                        gr.Audio(source="microphone", label="5. 小朋友, 你们不要再欺负它了"),
                    ]
                with gr.Column(scale=1):
                    audio_lst2 = [
                        gr.Audio(source="microphone", label="6. 可是, 小黄鸭并不怕他们"),
                        gr.Audio(source="microphone", label="7. 然后, 他们一起走了很长一段时间"),
                        gr.Audio(source="microphone", label="8. 突然, 墙壁后面传来一阵声音"),
                        gr.Audio(source="microphone", label="9. 结果盘子掉在地上, 打得粉碎"),
                        gr.Audio(source="microphone", label="10. 四个小伙伴很开心, 一起感谢小松鼠的帮助"),
                    ]
                with gr.Column(scale=1):
                    audio_lst3 = [
                        gr.Audio(
                            source="microphone", label="11. 不过, 当他看到拇指姑娘的时候, 他马上就变得高兴起来"
                        ),
                        gr.Audio(source="microphone", label="12. 从此以后, 他过上了幸福的生活"),
                        gr.Audio(source="microphone", label="13. 老山羊最后伤心地, 哭着走了出去"),
                        gr.Audio(source="microphone", label="14. 而且准备一直找下去, 直到他走不动为止"),
                        gr.Audio(source="microphone", label="15. 海马先生轻轻游过大海"),
                    ]
                with gr.Column(scale=1):
                    audio_lst4 = [
                        gr.Audio(
                            source="microphone", label="16. 一起高高兴兴地, 回到了他们的爸爸妈妈身边"
                        ),
                        gr.Audio(
                            source="microphone", label="17. 艾丽莎很小不能去上学, 但她有一个非常贵重精美的画册"
                        ),
                        gr.Audio(source="microphone", label="18. 狮子还是够不着, 它叫来了狐狸"),
                        gr.Audio(
                            source="microphone", label="19. 姑娘坐到国王的马车上, 和国王一起回到宫中"
                        ),
                        gr.Audio(source="microphone", label="20. 温妮大叫了起来, 现在我们该怎么回家呀"),
                    ]
            # One task at a time
            train_progress = gr.Textbox(
                label="训练进度", value="当前无训练任务", interactive=False
            )
            with gr.Row():
                use_name = gr.Textbox(label="请输入克隆声音的名字(英文小写)", value="test", interactive=True)
                training_button = gr.Button("开始训练")

        with gr.TabItem("\N{party popper}模型体验") as tab_infer:
            dbg_output = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(
                        label="合成内容", value="这周天气真不错, 叫上朋友一起爬山吧", max_lines=3
                    )
                    # model_name = gr.Textbox(
                    #     label="模型名称",
                    #     value="model/speech_clone/janchen",
                    #     interactive=False,
                    # )

                    model_list = os.listdir("model/speech_clone")
                    logger.info(model_list)
                    
                    model_name = gr.CheckboxGroup(choices=model_list)
                    # out_wav_file_path = gr.Textbox(
                    #     label="输出文件路径", value="result/output.wav", interactive=False
                    # )
                    #  hot_models = gr.Radio(label="热门模型", choices=HOT_MODELS, type="value")
                    # user_models = gr.Radio(label="模型选择", choices=HOT_MODELS, type="value", value=HOT_MODELS[0])
                    # refresh_button = gr.Button("刷新模型列表")
                #  voice_name = gr.Label(label="当前选择模型", value='')
                with gr.Column(scale=1):
                    infer_progress = gr.Textbox(
                        label="合成进度", value="当前无合成任务", interactive=False
                    )
                    helper2 = gr.Markdown(
                        """
                    \N{police cars revolving light}**注意**:
                    本录音由AI生成, 禁止用于非法用途
                    """
                    )
                    audio_output = gr.Audio(label="合成结果")
                    inference_button = gr.Button("合成")

    #     user_models.change(fn=choice_user_voice, inputs=[user_models, session], outputs=session)

    # refresh_button.click(fetch_uuid, inputs=[uuid_txt, session], outputs=[dbg_output, session, user_models])

    audio_list = audio_lst1 + audio_lst2 + audio_lst3 + audio_lst4
    training_button.click(
        launch_training_task, inputs=[use_name] + audio_list, outputs=[train_progress]
    )

    inference_button.click(
        launch_infer_task,
        inputs=[model_name, text_input],
        outputs=[audio_output],
    )


# 生成自签名SSL证书
# ssl_cert = "certificate.pem"
# ssl_key = "private_key.pem"
# context = ssl.SSLContext(ssl.PROTOCOL_TLS)
# context.load_cert_chain(ssl_cert, ssl_key)

demo.queue(concurrency_count=20).launch(
    server_name="0.0.0.0",
    server_port=6006,
    # ssl_keyfile="private_key.pem",
    # ssl_certfile="certificate.pem",
    # ssl_verify=False,
)
