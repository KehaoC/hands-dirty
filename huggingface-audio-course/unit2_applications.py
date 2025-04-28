from datasets import load_dataset, Audio
from transformers import pipeline 
import json
import os
from dotenv import load_dotenv
import shutil
from transformers.utils import TRANSFORMERS_CACHE as DEFAULT_CACHE

def clear_cache():
    # 清理默认缓存目录
    default_cache = os.path.expanduser('~/.cache/huggingface')
    if os.path.exists(default_cache):
        print(f"清理默认缓存目录: {default_cache}")
        shutil.rmtree(default_cache)
    
    # 清理当前设置的缓存目录
    custom_cache = os.environ.get("TRANSFORMERS_CACHE")
    if custom_cache and os.path.exists(custom_cache):
        print(f"清理自定义缓存目录: {custom_cache}")
        shutil.rmtree(custom_cache)

def print_cache_info():
    print("当前缓存配置:")
    print("HF_HOME:", os.environ.get("HF_HOME"))
    print("HF_DATASETS_CACHE:", os.environ.get("HF_DATASETS_CACHE"))
    print("TRANSFORMERS_CACHE:", os.environ.get("TRANSFORMERS_CACHE"))
    print("默认缓存目录:", DEFAULT_CACHE)

def classification(minds):
    # 加载模型
    classifier = pipeline(
        "audio-classification",
        model="anton-l/xtreme_s_xlsr_300m_minds14",
        model_kwargs={"cache_dir": "../cache/models"}
    )
    example = minds[0]
    result = classifier(example["audio"]["array"])
    print("模型输出：\n", json.dumps(result, indent=4))

    id2label = minds.features["intent_class"].int2str
    print(" 正确标签：\n", id2label(example["intent_class"]))
    pass

def speech_recognition(minds):
    # 定义模型
    print("开始下载模型...")
    asr = pipeline(
        "automatic-speech-recognition",
        model="maxidl/wav2vec2-large-xlsr-german",
        model_kwargs={"cache_dir": "../cache/models"}
    )
    print("模型下载完成")
    example = minds[0]
    print("原文：", example["transcription"])
    
    result = asr(example["audio"]["array"])
    print("Rsult: ", result)
    # 模型运行


    pass

if __name__ == "__main__":
    # 定义数据集
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train", cache_dir="../cache/datasets")
    minds_for_speech_recognition = load_dataset("PolyAI/minds14", name="de-DE", split="train", cache_dir="../cache/datasets")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

    # classification(minds)
    speech_recognition(minds_for_speech_recognition)