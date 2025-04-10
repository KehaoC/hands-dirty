def behindPipeline():
    """https://huggingface.co/learn/llm-course/chapter2/2?fw=pt"""
    from transformers import AutoTokenizer
    from transformers import AutoModelForSequenceClassification
    import torch

    # 定义Tokenizer
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    raw_inputs = [
        "I've been waiting for a Hugging Face course my whole life.",
        "I hate this so much!",
    ]

    # 获取 token_id 序列
    inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
    # print(inputs)

    # 获取模型
    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    # 获取模型输出
    output = model(**inputs)
    # 等价于 model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

    predictions = torch.nn.functional.softmax(output.logits, dim=-1)
    print(predictions)
    print(model.config.id2label)

def save_models():
    """https://huggingface.co/learn/llm-course/chapter2/3?fw=pt"""

    from transformers import BertConfig, BertModel

    # 获得配置
    config = BertConfig()
    print("Config:\n", config)

    # 加载模型配置, 是随意的一个模型
    model = BertModel(config)

    model = BertModel.from_pretrained("bert-base-uncased")

    model.save_pretrained("../pretrained_models/")

def use_models_to_inference():
    from transformers import AutoTokenizer
    sequences = ["Hello!", "Cool.", "Nice!"]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokenizer_output = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")

    encoded_sequences = tokenizer_output["input_ids"]
    print(encoded_sequences)

    # 实际上上一步已经得到了 pytorch 类型的张量
    import torch
    model_inputs = torch.tensor(encoded_sequences)
    print(model_inputs)

def build_tokenizer():
    """
    Tokenizer: To translate text into tokens that could be processed by the model. 
    """
    from transformers import BertTokenizer, AutoTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    raw_text = "Using a Transformer network is simple"
    tokenizer_output = tokenizer(raw_text)
    print(tokenizer_output)

    # 存储模型
    # tokenizer.save_pretrained("../pretrained_models/tokenizer")

    # 拆分步骤来看, 第一步是切分 token
    tokens = tokenizer.tokenize(raw_text)
    print(tokens)

    # 第二步是把token 转化为 id
    ids = tokenizer.convert_tokens_to_ids(tokens)
    print(ids)

    # 再试试解码
    decoded_string = tokenizer.decode(ids)
    print(decoded_string)

def handling_multiple_sequences():
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence = "I've been waiting for a hugginface course my whole life."

    tokens = tokenizer.tokenize(sequence)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    batched_ids = [ids, ids]
    print(batched_ids)

    inputs_ids_bugs = torch.tensor(batched_ids)
    print(inputs_ids_bugs)
    inputs_ids = torch.tensor([batched_ids])
    print(inputs_ids)

    output = model(inputs_ids_bugs)
    print(output.logits)

    print("============")
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequence1_ids = [[200, 200, 200]]
    sequence2_ids = [[200, 200]]

    batched_ids = [
        [200, 200, 200],
        [200, 200, tokenizer.pad_token_id]
    ]

    print(model(torch.tensor(sequence1_ids)).logits)
    print(model(torch.tensor(sequence2_ids)).logits)
    print(model(torch.tensor(batched_ids)).logits)

    print("========Attention Mast=============")
    attention_mask = [
        [1, 1, 1],
        [1, 1, 0],
    ]

    outputs = model(torch.tensor(batched_ids), attention_mask=torch.tensor(attention_mask))
    print(outputs.logits)

    print("========Small Test=============")
    sequence1 = "I've been waiting for a HuggingFace course my whole life."
    sequence2 = "I hate this so much!"
    sequences = [sequence1, sequence2]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    ids1 = tokenizer(sequence1)
    ids2 = tokenizer(sequence2)
    ids = tokenizer(sequences, padding=True)
    print(ids)

    print(model(torch.tensor([ids1['input_ids']])).logits)
    print(model(torch.tensor([ids2['input_ids']])).logits)
    print(model(torch.tensor(ids['input_ids']), torch.tensor(ids['attention_mask'])).logits)

def putting_all_together():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

    tokens = tokenizer(sequences, padding="longest", truncation=True, return_tensors="pt")
    output = model(**tokens)
    print(output.logits)







if __name__ == "__main__":
    # behindPipeline()
    # save_models()
    # use_models_to_inference()
    # build_tokenizer()
    # handling_multiple_sequences()
    putting_all_together()
