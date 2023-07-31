import requests
import json
from loguru import logger

url = "http://127.0.0.1:8501"


def get_data_label(
    input_wav="data/speech_clone/janchen", 
    output_data="data/speech_clone/janchen_label"
):
    data = {
        "input_wav": input_wav,
        "output_data": output_data,
    }

    api_url = url + "/label_data"
    response = requests.post(api_url, data=json.dumps(data))
    logger.info(response.text)


def get_data_train(
    output_data="data/speech_clone/janchen_label",
    model_name="model/speech_clone/janchen",
):
    data = {"output_data": output_data, "model_name": model_name}

    api_url = url + "/train_data"
    response = requests.post(api_url, data=json.dumps(data))
    logger.info(response.text)


def get_data_infer(
    model_name="model/speech_clone/janchen",
    content="今天的天气真不错",
    output_wav_path="output.wav",
):
    data = {"model_name": model_name, "content": content, "output_wav": output_wav_path}

    api_url = url + "/infer_data"
    response = requests.post(api_url, data=json.dumps(data))
    logger.info(response.text)

    # return response.json()["output_wav"]


if __name__ == "__main__":
    # get_data_label()
    # get_data_train()
    get_data_infer()
