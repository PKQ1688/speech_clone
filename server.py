import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from loguru import logger

from ms_main_pipe import get_label_data, train_model, model_infer


app = FastAPI()


class DataPath(BaseModel):
    input_wav: str = Field(default="data/speech_clone/janchen", description="输入音频")
    output_data: str = Field(
        default="data/speech_clone/janchen_label", description="输出音频"
    )
    model_name: str = Field(default="model/speech_clone/janchen", description="用户模型路径")
    output_wav: str = Field(default="output.wav", description="输出音频")
    content: str = Field(default="今天的天气真不错", description="推理文本")


@app.post("/label_data", summary="用于标注用户自己提供的音频数据")
def _label_data(data_path: DataPath):
    get_label_data(input_wav=data_path.input_wav, output_data=data_path.output_data)
    return {"code": 200, "msg": "标注成功"}


@app.post("/train_data", summary="训练用户自己提供的音频数据")
def _train_data(data_path: DataPath):
    train_model(output_data=data_path.output_data, user_name=data_path.model_name)
    return {"code": 200, "msg": "训练成功"}


@app.post("/infer_data", summary="使用用户自己的模型进行推理")
def _infer_data(data_path: DataPath):
    model_infer(
        model_dir=data_path.model_name,
        content=data_path.content,
        output_wav=data_path.output_wav,
    )
    return {"code": 200, "msg": "推理成功"}


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8501, reload=False, workers=0)
