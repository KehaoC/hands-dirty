def startup():
    import torch
    from torch.optim import AdamW
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

    sequences = [
        "I've been waiting for a HuggingFace course my whole life.",
        "This course is amazing!"
    ]

    # 转换为 tokens 的相关 objects
    batch = tokenizer(sequences, truncation=True, padding=True, return_tensors="pt")

    batch["labels"] = torch.tensor([1, 1])

    optimizer = AdamW(model.parameters())
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()

def loading_datasets():
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # 获取数据集
    raw_datasets = load_dataset("glue", "mrpc")
    raw_train_dataset = raw_datasets["train"]

    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # tokenized_sentence_1 = tokenizer(raw_datasets['train']['sentence1'])
    # tokenized_sentence_2 = tokenizer(raw_datasets['train']['sentence2'])

    # inputs = tokenizer("This is the first sentence.", "This is the second one.")
    # print(tokenizer.convert_ids_to_tokens(inputs["input_ids"]))
    # print(tokenizer.decode(inputs["input_ids"]))

    #  方法 1：tokenize整个数据集
    # tokenized_dataset = tokenizer(
    #     raw_datasets["train"]["sentence1"],
    #     raw_datasets["train"]["sentence2"],
    #     padding=True,
    #     truncation=True
    # )

    # 方法2：使用函数加map, 效率更优，占用峰值内存更小
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    import time

    num_proc = 1  # 这里有点奇怪，为什么num_proc = 1 的时候效率远高于其他线程数目？
    start_time = time.time()
    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True, num_proc=num_proc)
    end_time = time.time()

    print(f"Tokenization completed in {end_time - start_time:.2f} seconds. Number of Process = {num_proc}")

    from transformers import DataCollatorWithPadding

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    samples = tokenized_dataset["train"][:8]
    # print(samples)

    # 去掉 ids, sentence1, sentence2
    samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
    
    # 查看每个sample的 tokenize 之后的 input_ids的长度
    print([len(x) for x in samples["input_ids"]])

    batch = data_collator(samples)
    print({k: v.shape for k, v in batch.items()})

def trainer():
    import torch
    if torch.backends.mps.is_available():
        print("MPS is available.")
    else:
        print("CPU is using.")

    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    print("Loading dataset...")
    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    print("Tokenizing datasets...")
    tokenized_dataset = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 设置评估, 老写法
    # print("Starting evaluating...")
    # predictions = trainer.predict(tokenized_dataset["validation"])
    # print(predictions.predictions.shape, predictions.label_ids.shape)

    # import numpy as np
    # preds = np.argmax(predictions.predictions, axis=-1)

    # import evaluate
    # metric = evaluate.load("glue", "mrpc")
    # print(metric.compute(predictions=preds, references=predictions.label_ids))

    # 设置评估，transformer写法
    import evaluate
    import numpy as np
    def compute_metrics(eval_preds):
        metric = evaluate.load("glue", "mrpc")
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)
        
        # 前者是预测值，后者是真实值
        return metric.compute(predictions=predictions, references=labels)
    
    from transformers import TrainingArguments, AutoModelForSequenceClassification, Trainer
    training_args = TrainingArguments(
        "../pretrained_models/test-trainer",
        evaluation_strategy="epoch",
        num_train_epochs=1,
    )

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    trainer = Trainer(
        model,
        training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    print("Start training...")
    trainer.train()

    pass
if __name__ == "__main__":
    # startup()
    # loading_datasets()
    trainer()