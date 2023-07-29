import gradio as gr

with gr.Blocks(css="#warning {color: red} .feedback {font-size: 24px}") as demo:
    gr.Markdown("## 灵动树上声音克隆")
    voice: str = ''
    # model_list = PRETRAINED_MODEL_ID
    my_models = []
    session = gr.State({
        'modelscope_uuid': 'guest',
        'modelscope_request_id': 'test',
        'voice': voice,
        # 'model_list': model_list,
        'my_models': my_models
    })

    with gr.Tabs():
        with gr.TabItem("\N{rocket}模型定制") as tab_train:
            helper = gr.Markdown(
            """
            \N{glowing star}**定制步骤**\N{glowing star}

            Step 1. 录制音频\N{microphone}，点击下方音频录制并朗读左上角文字, 请至少录制10句话

            Step 2. 点击 **[开始训练]** \N{hourglass with flowing sand}，启动模型训练，等待约10分钟

            Step 3. 切换至 **[模型体验]** \N{speaker with three sound waves}，选择训练好的模型，感受效果

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
                        gr.Audio(source="microphone", label="2. 不行, 他想了一下, 我不能这样对国王说, 这是在撒谎"),
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
                        gr.Audio(source="microphone", label="11. 不过, 当他看到拇指姑娘的时候, 他马上就变得高兴起来"),
                        gr.Audio(source="microphone", label="12. 从此以后, 他过上了幸福的生活"),
                        gr.Audio(source="microphone", label="13. 老山羊最后伤心地, 哭着走了出去"),
                        gr.Audio(source="microphone", label="14. 而且准备一直找下去, 直到他走不动为止"),
                        gr.Audio(source="microphone", label="15. 海马先生轻轻游过大海"),
                    ]
                with gr.Column(scale=1):
                    audio_lst4 = [
                        gr.Audio(source="microphone", label="16. 一起高高兴兴地, 回到了他们的爸爸妈妈身边"),
                        gr.Audio(source="microphone", label="17. 艾丽莎很小不能去上学, 但她有一个非常贵重精美的画册"),
                        gr.Audio(source="microphone", label="18. 狮子还是够不着, 它叫来了狐狸"),
                        gr.Audio(source="microphone", label="19. 姑娘坐到国王的马车上, 和国王一起回到宫中"),
                        gr.Audio(source="microphone", label="20. 温妮大叫了起来, 现在我们该怎么回家呀"),
                    ]
            # One task at a time
            train_progress = gr.Textbox(label="训练进度", value="当前无训练任务", interactive=False)
            with gr.Row():
                training_button = gr.Button("开始训练")

        with gr.TabItem("\N{party popper}模型体验") as tab_infer:
            uuid_txt = gr.Text(label="modelscope_uuid", visible=False)
            reqid_txt = gr.Text(label="modelscope_request_id", visible=False)
            dbg_output = gr.Textbox(visible=False)

            with gr.Row():
                with gr.Column(scale=1):
                    text_input = gr.Textbox(label="合成内容", value="这周天气真不错, 叫上朋友一起爬山吧", max_lines=3)
                    #  hot_models = gr.Radio(label="热门模型", choices=HOT_MODELS, type="value")
                    # user_models = gr.Radio(label="模型选择", choices=HOT_MODELS, type="value", value=HOT_MODELS[0])
                    # refresh_button = gr.Button("刷新模型列表")
                #  voice_name = gr.Label(label="当前选择模型", value='')
                with gr.Column(scale=1):
                    infer_progress = gr.Textbox(label="合成进度", value="当前无合成任务", interactive=False)
                    helper2 = gr.Markdown(
                    """
                    \N{bell}**温馨提示**:
                    点击 **[刷新模型列表]** 拉取已定制的模型, 首次合成会下载模型, 请耐心等待

                    \N{police cars revolving light}**注意**:
                    本录音由AI生成, 禁止用于非法用途
                    """
                    )
                    audio_output = gr.Audio(label="合成结果")
                    inference_button = gr.Button("合成")

    #     user_models.change(fn=choice_user_voice, inputs=[user_models, session], outputs=session)

    # refresh_button.click(fetch_uuid, inputs=[uuid_txt, session], outputs=[dbg_output, session, user_models])

    audio_list = audio_lst1 + audio_lst2 + audio_lst3 + audio_lst4
    # training_button.click(launch_training_task,
                        #   inputs=[uuid_txt, session] + audio_list,
                        #   outputs=[train_progress, user_models, session])

    # inference_button.click(launch_infer_task,
    #                        inputs=[uuid_txt, user_models, text_input, session],
    #                        outputs=[infer_progress, audio_output])


demo.queue(concurrency_count=20).launch(server_name="0.0.0.0",server_port=6006)