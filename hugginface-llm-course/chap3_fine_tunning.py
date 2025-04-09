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

def pytorch_train():
    print("Loading libraries...")
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    print("Loading raw datasets...")
    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    print("Tokenizing datasets...")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 数据集处理步骤
    print("Post-Processing datasets...")

    # Features：['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask']
    print(tokenized_datasets)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    print(tokenized_datasets["train"].column_names)

    # 定义DataLoader
    print("Desiging data loader from troch...")
    from torch.utils.data import DataLoader
    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    # 检查一下数据加载结果
    for batch in train_dataloader:
        print({k: v.shape for k, v in batch.items()})
        break

    # 定义模型
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    print("Getting model outputs test...")
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e-5)

    from transformers import get_scheduler

    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )
    print("Num training steps is: ", num_training_steps)

    import torch
    device = torch.device("mps") if torch.cuda.is_available else torch.device("mps")
    model.to(device)
    print("Using device: ", device)

    from tqdm.auto import tqdm
    progress_bar = tqdm(range(num_training_steps))

    print("开始训练：")
    model.train()
    for epoch in num_epochs:
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    import evaluate

    metric = evaluate.load("glue", "mrpc")
    model.eval()
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    metric.compute()
    pass

def without_accelerate():
    import time
    total_start_time = time.time()
    
    # 1. 加载数据集
    print("加载数据集中...")
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # 2. Tokenize and post-process datasets
    print("后处理以及 tokenize 数据集...")
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    from torch.utils.data import DataLoader
    train_size = int(len(tokenized_datasets["train"])*0.01)
    train_dataloader = DataLoader(
        tokenized_datasets["train"].select(range(train_size)), shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    print("加载模型...")
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e05)

    from transformers import get_scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    print("设置设备...")
    import torch
    device = torch.device("mps")
    model.to(device)
    
    print("开始训练")
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(num_training_steps))

    train_start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    train_end_time = time.time()
    print(f"\n训练耗时: {train_end_time - train_start_time:.2f} 秒")

    print("开始测试...")
    eval_start_time = time.time()
    import evaluate
    metric = evaluate.load("glue", "mrpc")
    model.eval()

    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    result = metric.compute()
    eval_end_time = time.time()
    print(f"评估耗时: {eval_end_time - eval_start_time:.2f} 秒")
    print("评估结果: ", result)
    
    total_end_time = time.time()
    print(f"\n总耗时: {total_end_time - total_start_time:.2f} 秒")

def with_accelerate():
    import time
    total_start_time = time.time()

    from accelerate import Accelerator
    accelerator = Accelerator()
    
    # 1. 加载数据集
    print("加载数据集中...")
    from datasets import load_dataset
    from transformers import AutoTokenizer, DataCollatorWithPadding

    raw_datasets = load_dataset("glue", "mrpc")
    checkpoint = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)

    # 2. Tokenize and post-process datasets
    print("后处理以及 tokenize 数据集...")
    def tokenize_function(example):
        return tokenizer(example["sentence1"], example["sentence2"], truncation=True)
    
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")

    from torch.utils.data import DataLoader
    train_size = int(len(tokenized_datasets["train"])*0.01)
    train_dataloader = DataLoader(
        tokenized_datasets["train"].select(range(train_size)), shuffle=True, batch_size=8, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
    )

    print("加载模型...")
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

    from torch.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=5e05)

    from transformers import get_scheduler
    num_epochs = 3
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    print("设置设备...")
    import torch
    # device = torch.device("mps")
    # model.to(device)

    train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, eval_dataloader, model, optimizer
    )
    
    print("开始训练")
    from tqdm.auto import tqdm
    progress_bar = tqdm(range(num_training_steps))

    train_start_time = time.time()
    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
    
    train_end_time = time.time()
    print(f"\n训练耗时: {train_end_time - train_start_time:.2f} 秒")

    print("开始测试...")
    eval_start_time = time.time()
    import evaluate
    metric = evaluate.load("glue", "mrpc")
    model.eval()

    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    
    result = metric.compute()
    eval_end_time = time.time()
    print(f"评估耗时: {eval_end_time - eval_start_time:.2f} 秒")
    print("评估结果: ", result)
    
    total_end_time = time.time()
    print(f"\n总耗时: {total_end_time - total_start_time:.2f} 秒")
    pass

if __name__ == "__main__":
    # startup()
    # loading_datasets()
    # trainer()
    # pytorch_train()
    
    # without_accelerate()
    with_accelerate()