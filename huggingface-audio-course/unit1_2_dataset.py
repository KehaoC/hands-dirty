def load_and_explore():
    from datasets import load_dataset

    minds = load_dataset("PolyAI/minds14", name="en-AU", split="train")
    print("数据集概览：", minds)
    example = minds[0]
    print("样例：", example)
    ids2label = minds.features["intent_class"].int2str
    example_label = ids2label(example["intent_class"])
    print("样例类别：", example_label)

    columns_to_remove = ["lang_id", "english_transcription"]
    minds = minds.remove_columns(columns_to_remove)
    print("清除无用标签后的数据集: ", minds)

    import gradio as gr

    def generate_audio():
        example = minds.shuffle()[0]
        audio = example["audio"]
        return (
            audio["sampling_rate"],
            audio["array"]
        ), ids2label(example["intent_class"])
    
    with gr.Blocks() as demo:
        with gr.Column():
            for _ in range(4):
                audio, label = generate_audio()
                output = gr.Audio(audio, label=label)
    
    demo.launch(debug=True)

    # 显示出来
    import librosa
    import matplotlib.pyplot as plt
    import librosa.display

    array = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    plt.figure().set_figwidth(12)
    librosa.display.waveshow(array, sr=sampling_rate)
    plt.show()

if __name__ == "__main__":
    load_and_explore()