
dataset_name = "PolyAI/minds14"
def resampling():
    print("加载数据集...\n")
    from datasets import Audio, load_dataset
    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    pass

def filtering():
    from datasets import load_dataset, Audio
    import librosa
    minds = load_dataset(dataset_name, name="en-AU", split="train")
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))

    MAX_DURATION_IN_SECONDS = 20.0

    def is_audio_length_in_range(input_length):
        return input_length < MAX_DURATION_IN_SECONDS

    # 添加一个新列，用于计算时间
    new_column = [librosa.get_duration(path=x) for x in minds["path"]]
    minds = minds.add_column("duration", new_column)

    minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])

    minds = minds.remove_columns(["duration"])
    print(minds)
    pass

def pre_processing():
    from transformers import WhisperFeatureExtractor
    from datasets import load_dataset, Audio
    import librosa

    minds = load_dataset(dataset_name, name="en-AU", split="train")
    feature_extractor = WhisperFeatureExtractor.from_pretrained("openai/whisper-small")

    # 调整采样率
    minds = minds.cast_column("audio", Audio(sampling_rate=16_000))
    
    # 过滤数据
    MAX_DURATION = 20.0
    def is_audio_length_in_range(input_length):
        return input_length<MAX_DURATION
    new_column = [librosa.get_duration(path=x) for x in minds["path"]]
    minds = minds.add_column("duration", new_column)
    minds = minds.filter(is_audio_length_in_range, input_columns=["duration"])
    minds = minds.remove_columns(["duration"])

    def prepare_dataset(example):
        # 提取高级特征
        audio = example["audio"]
        features = feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"], padding=True
        )
        return features
    
    minds = minds.map(prepare_dataset)
    print(minds) 

    import numpy as np
    import matplotlib.pyplot as plt

    example = minds[0]
    input_features = example["input_features"]

    plt.figure().set_figwidth(12)
    librosa.display.specshow(
        np.asarray(input_features[0]),
        x_axis="time",
        y_axis="mel",
        sr=feature_extractor.sampling_rate,
        hop_length=feature_extractor.hop_length,
    )
    plt.colorbar()
    plt.show()

    pass

if __name__ == "__main__":
    # resampling()
    # filtering()
    pre_processing()