import os
import json
import random
from itertools import chain
from functools import partial

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments, DataCollatorForTokenClassification
from tokenizers import AddedToken
import evaluate
from datasets import Dataset
import numpy as np


def filter_no_pii(example, percent_allow=0.2):
    # Return True if there is PII
    # Or 20% of the time if there isn't
    # To remove 80% of entry that only has "O" in the labels
    import random
    has_pii = set("O") != set(example["provided_labels"])

    return has_pii or (random.random() < percent_allow)

def tokenize(example, tokenizer, label2id, max_length):
    import numpy as np
    from tokenizers import AddedToken
    text = []
    labels = []

    for t, l, ws in zip(example["tokens"], example["provided_labels"], example["trailing_whitespace"]):

        text.append(t)
        labels.extend([l]*len(t))
        if ws:
            text.append(" ")
            labels.append("O")

    tokenized = tokenizer("".join(text), return_offsets_mapping=True, max_length=max_length)

    labels = np.array(labels)

    text = "".join(text)
    token_labels = []

    for start_idx, end_idx in tokenized.offset_mapping:

        # CLS token
        if start_idx == 0 and end_idx == 0:
            token_labels.append(label2id["O"])
            continue

        # case when token starts with whitespace
        if text[start_idx].isspace():
            start_idx += 1

        while start_idx >= len(labels):
            start_idx -= 1

        token_labels.append(label2id[labels[start_idx]])

    length = len(tokenized.input_ids)

    return {
        **tokenized,
        "labels": token_labels,
        "length": length
    }

def tokenize_and_align_labels(examples, tokenizer, label2id, max_length):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True, max_length=max_length)
    
    labels = []
    for i, label in enumerate(examples["provided_labels"]):
        print("##### Label %d is %s #####" %(i,label))
        word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
        print("Word_ids list: ")
        print(word_ids)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label2id[label[word_idx]])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        print("label_ids: ")
        print(label_ids)
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p, metric, all_labels):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [all_labels[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [all_labels[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)

    # Unpack nested dictionaries
    final_results = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for n, v in value.items():
                final_results[f"{key}_{n}"] = v
        else:
            final_results[key] = value
    return final_results

def main(training, training_model_path, training_max_length, path_to_datasets, model_save_path):
    data = []
    # append every entry row of data (dictionary) into a list
    for dataset in path_to_datasets:
      data.extend(json.load(open(dataset)))

    print(type(data)) # == list
    print(type(data[0])) # == dict
    print((data[0]))
    print(type(data[-1])) # == dict
    print((data[-1]))
    # print((data[1]))
    # print((data[2]))
    print("Total number of data entries: ", len(data))

    # list of all different types of labels
    all_labels = sorted(list(set(chain(*[x["labels"] for x in data]))))

    # dictionary of label (key) to id (value) pair
    label2id = {l: i for i,l in enumerate(all_labels)}
    print("All label 2 id: ", label2id)
    # dictionary of id (key) to label (value) pair
    id2label = {v:k for k,v in label2id.items()}

    # input data into Dataset object, where ds[0] is the dictionary of the first entry row
    ds = Dataset.from_dict({
        "full_text": [x["full_text"] for x in data],
        "document": [str(x["document"]) for x in data],
        "tokens": [x["tokens"] for x in data],
        "trailing_whitespace": [x["trailing_whitespace"] for x in data],
        "provided_labels": [x["labels"] for x in data]
    })

    # load tokenizer type used by the pretrained model
    tokenizer = AutoTokenizer.from_pretrained(training_model_path)

    # tokens are only added if they are not already in the vocabulary. Add a AddedToken or list of str / AddedToken
    # lots of newlines in the text
    tokenizer.add_tokens(AddedToken("\n", normalized=False))
    
    ds = ds.filter(
        filter_no_pii,
        num_proc=2,
    )

    # To curate total number of each PIIs again

    ds = ds.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer, "label2id": label2id, "max_length": training_max_length},
        num_proc=2,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=16)

    metric = evaluate.load("seqeval")

    # define base model to be used (can be transfer learning from existing model) and; 
    # define the possible output labels (y) that model should predict, and the id of these labels
    model = AutoModelForTokenClassification.from_pretrained(training_model_path, num_labels=len(all_labels), id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

    # when adding new tokens to the vocabulary, you should make sure to also resize the token embedding matrix of the model so that its embedding matrix matches the tokenizer
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=16)

    ### All possible training arguments that may be defined in may be found here: (To-explore)
    # https://huggingface.co/docs/transformers/v4.39.2/en/main_classes/trainer#transformers.TrainingArguments 
    # https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
    args = TrainingArguments(
        output_dir="output",
        fp16=True, # automatic fixed precision (AMP)
        learning_rate=2.5e-5,
        lr_scheduler_type="linear",
        weight_decay=0.01,
        warmup_ratio=0.1,
        per_device_train_batch_size=2, # effective_batch_size = 16, train_batch_size * gradient_accum_steps
        per_device_eval_batch_size=2,
        report_to="tensorboard",
        evaluation_strategy="epoch",
        # eval_steps=200, # only when evaluation_strategy is steps, to sync with save_steps
        logging_strategy="steps",
        save_strategy="epoch",
        # save_steps=200, # only when save_strategy is steps
        # seed=1,
        save_total_limit=2, # if set to 1, this makes load_best_model_at_end redundant
        load_best_model_at_end=True,
        logging_steps=5,
        metric_for_best_model="overall_recall",
        greater_is_better=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        dataloader_num_workers=1,
        logging_dir="output_log"
    )
    ### About training a better performing and scalable model and about effective batch size: https://huggingface.co/docs/transformers/v4.18.0/en/performance

    # split dataset into train and test splits
    final_ds = ds.train_test_split(test_size=0.2) # shuffle is true by default

    print(model)
    print("_________________________________________________")

    ### How to freeze model layers  (https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6)
    # modules = [model.deberta.embeddings, model.deberta.encoder.layer[:6]] # Freeze first 6 transformer layers
    modules = [model.deberta.encoder.layer[:6]]
    for module in modules:
        for param in module.parameters():
            param.requires_grad = False
    # for name, param in model.named_parameters():
    #  if name.startswith("..."): # choose whatever name you like here
    #     param.requires_grad = False

    # Example to freeze all layers except the classification layer 
    # for param in model.bert.bert.parameters():
        # param.requires_grad = False
    ## if freeze classification layer too
    # for param in model.bert.parameters():
    #     param.requires_grad = False

    ### to verify
    for name, param in model.named_parameters():
        print(name, param.requires_grad)
    print("_________________________________________________")
    ### for specific param freezing 
    # params = model.state_dict()
    # print(params.keys()) # check existing params keys
    # For example: model.fc1.weight.requires_grad = False

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=final_ds["train"],
        eval_dataset=final_ds["test"],
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=partial(compute_metrics, metric=metric, all_labels=all_labels),
    )

    print("Begin training...")
    trainer.train()
    # trainer.save_model(model_save_path) # saves tokenizer with model
    trainer.save_state()
    model.save_pretrained(args.output_dir)
    
    return


if __name__ == "__main__":
    training = True # be sure to turn internet off if doing inference

    training_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model\deberta-v3-base"
    # training_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model\deberta-v3-base"
    # training_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model\model_2024Mar26_123\checkpoint-1965"
    training_max_length = 3500 # 3072 # 512

    # inference_model_path = "/kaggle/input/pii-data-detection-baseline/output/checkpoint-240" # replace with our own trained model
    inference_model_path = r"C:\Users\abome\source\repos\venv_pii_ner\model"
    inference_max_length = 3500 # 512

    # path_to_datasets = [r'C:\Users\abome\source\repos\venv_pii_ner\dataset\train.json'])
    path_to_datasets = [r'C:\Users\abome\source\repos\venv_pii_ner\dataset\train.json', 
                        r'C:\Users\abome\source\repos\venv_pii_ner\dataset\pii_dataset_fixed.json',
                        r'C:\Users\abome\source\repos\venv_pii_ner\dataset\mpware_mixtral8x7b_v1.1-no-i-username.json',
                        r'C:\Users\abome\source\repos\venv_pii_ner\dataset\moredata_dataset_fixed.json',
                        ] # append path to other dataset json files to this list)
    model_save_path = r'C:\Users\abome\source\repos\venv_pii_ner\model\\'

    main(training, training_model_path, training_max_length, path_to_datasets, model_save_path)
