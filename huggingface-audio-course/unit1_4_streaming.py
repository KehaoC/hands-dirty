def steaming():
    import json
    from datasets import load_dataset
    gigaspeech = load_dataset("speechcolab/gigaspeech", "xs", streaming=True)
    example = next(iter(gigaspeech["train"]))
    print(json.dumps(example, indent=4))

    pass

if __name__ == "__main__":
    steaming()