import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class GLUEDataModule(pl.LightningDataModule):
    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]
    
    def __init__(self, model_name, task_name, max_seq_length=128, batch_size=32, num_workers=7):
        super().__init__()
        self.model_name = model_name
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def setup(self, stage=None):
        # Load dataset
        print(self.task_name)
        self.dataset = load_dataset("glue", self.task_name, 
                    cache_dir="/local/scratch/yc538/.cache/huggingface/datasets")

        # Preprocess datasets
        for split in self.dataset.keys():
            # print(f"split = {split}")
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"]
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)

        # Set splits
        self.train_dataset = self.dataset["train"]
        self.val_dataset = self.dataset["validation_matched" if self.task_name == "mnli" else "validation"]
        self.test_dataset = self.dataset["test_matched" if self.task_name == "mnli" else "test"]
        
    
    def convert_to_features(self, examples, indices=None):
        if self.task_name == "sst2":
            texts = examples['sentence']
        elif self.task_name == "mnli":
            texts = list(zip(examples['premise'], examples['hypothesis']))
        else: texts = list(zip(examples['question'], examples['sentence']))

        features = self.tokenizer.batch_encode_plus(texts, padding='max_length', 
                    truncation=True, max_length=self.max_seq_length)

        features["labels"] = examples["label"]

        return features
        
    def _tokenize(self, examples):
        # Tokenize the inputs and labels
        return self.tokenizer(examples['sentence'] if self.task_name == "sst2" 
                              else examples['premise'] + " " + examples['hypothesis'],
                              padding='max_length', truncation=True, max_length=self.max_seq_length)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)