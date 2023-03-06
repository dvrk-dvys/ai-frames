import os, psutil, timeit, re, math, argparse
import wandb
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from typing import Optional

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.accelerators import CPUAccelerator, GPUAccelerator
from pytorch_lightning.utilities.data import DataLoader as pl_dataLoader
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger

import seaborn as sns
# from pytorch_lightning.plugins.io import AsyncCheckpointIO
# from torchvision.transforms import ToTensor

import torch
from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Huggingface
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DistilBertTokenizerFast,
    DistilBertConfig,
    get_linear_schedule_with_warmup,
    EncoderDecoderModel)
from datasets import load_dataset  #, Dataset

# ax
# premise: a string feature.
# hypothesis: a string feature.
# label: a classification label, with possible values including entailment (0), neutral (1), contradiction (2).
# idx: a int32 feature.

# mnli_matched
# premise: a string feature.
# hypothesis: a string feature.
# label: a classification label, with possible values including entailment (0), neutral (1), contradiction (2).
# idx: a int32 feature.
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer):
        # self.raw_texts = data['text']
        # self.labels = data['label']
        self.data = data
        # self.classification_labels = data['label']
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        raw_text = self.data[idx]['text']
        label = self.data[idx]['label']

        # Tokenize the input text
        encoded_text = self.tokenizer.encode_plus(
            self.raw_texts[idx],
            add_special_tokens=True,
            max_length=512,
            padding='max_length',
            truncation=True
        )

        # Extract the input ids and attention mask from the tokenized input
        input_ids = encoded_text['input_ids']
        attention_mask = encoded_text['attention_mask']
        # the empty tensor at the end of the return is for the classification
        return torch.tensor(input_ids), torch.tensor(attention_mask), torch.tensor(self.labels[idx])  #, torch.tensor([0])

def SentimentCollate_fn(batch):
    """Function to create a mini-batch from a list of samples.
    Args:
        batch: A list of samples from the dataset.
    Returns:
        A mini-batch of tensors.
    """
    texts, attention_masks, sentiment = zip(*batch)
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    padded_sentiment = pad_sequence(sentiment, batch_first=True, padding_value=0)
    # padded_classification = torch.stack(classification)
    # return padded_texts, padded_attention_masks, padded_sentiment, padded_classification
    return {
        "input_ids": padded_texts,
        "attention_mask": padded_attention_masks,
        "sentiment": padded_sentiment,
    }

class SenModel(LightningModule):
    task_text_field_map = {
        # "cola": ["sentence"],
        # "sst2": ["sentence"],
        # "mrpc": ["sentence1", "sentence2"],
        # "qqp": ["question1", "question2"],
        # "stsb": ["sentence1", "sentence2"],
        "mnli": ["premise", "hypothesis"],
        # "qnli": ["question", "sentence"],
        # "rte": ["sentence1", "sentence2"],
        # "wnli": ["sentence1", "sentence2"],
        "ax": ["premise", "hypothesis"],
    }

    glue_task_num_labels = {
        # "cola": 2,
        # "sst2": 2,
        # "mrpc": 2,
        # "qqp": 2,
        # "stsb": 1,
        "mnli": 3,
        # "qnli": 2,
        # "rte": 2,
        # "wnli": 2,
        "ax": 3,
    }

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "mrpc",
        max_seq_length: int = 128,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size

        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.glue_task_num_labels[task_name]
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def __init__(self, num_classes, batch_size=32, num_workers=0):
        super().__init__()
        self.encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
        # self.classification_head = torch.nn.Linear(768, num_classes) # num_classes is the number of output classes
        # self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

        #!!! loading dataset directly into tensor is fastest
        self.dataset = Dataset.map(lambda examples: self.tokenizer(examples["text"], return_tensors="np"), batched=True)

        self.train_dataset = SentimentDataset(train_data, self.tokenizer)
        self.val_dataset = SentimentDataset(val_data, self.tokenizer)
        self.test_dataset = SentimentDataset(test_data, self.tokenizer)

        self.data_collator = SentimentCollate_fn
        self.train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator
        )
        self.val_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator
        )
        self.test_dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator
        )

        # self.classification_loss_fn = torch.nn.CrossEntropyLoss()
        self.sentiment_loss_fn = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder_decoder(input_ids, attention_mask=attention_mask, labels=labels)
        encoded_text = outputs.last_hidden_state
        sentiment = outputs.pooler_output
        # classification_pred = self.classification_head(sentiment)
        return {"encoded_text": encoded_text, "sentiment": sentiment}
    # The pooler output is a representation of the entire input sequence
    # that is produced by the final layer of the transformer in the BERT
    # model. The pooler output is a fixed-size vector that summarizes the
    # entire input sequence and is intended to be used as input to a
    # classifier or another downstream task.

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, sentiment = batch
        outputs = self.encoder_decoder(input_ids, attention_mask=attention_mask)
        encoded_text = outputs.last_hidden_state
        sentiment_pred = outputs.pooler_output
        sentiment_loss = self.sentiment_loss_fn(sentiment_pred, sentiment.float())
        self.log('train_sentiment_loss', sentiment_loss)
        return sentiment_loss

        # class probabilities for the input sequence.The sentiment
        # tensor that is passed as input to the classification_head is a
        # fixed-size representation of the entire input sequence that is produced
        # by the final layer of the BERT transformer model.The classification_head
        # is a linear layer that takes this representation as input and maps it to
        # a vector of size num_classes, where each element of the vector corresponds
        # to the probability of the input sequence belonging to a particular class.

        # You're right that in the case of binary sentiment classification, the sentiment prediction
        # and the classification task would be the same. Both tasks involve predicting a binary label
        # for a given input text - positive or negative. However, in some cases, the classification task
        # might involve predicting a multi-class label or a label that does not directly correspond to sentiment (e.g., topic classification, entity recognition, etc.).
        # In such cases, the classification_head would be trained to predict a vector of probabilities over all possible classes, and the predicted class would be the one with the highest probability.
        # In the SentimentModel implementation that you provided, the classification_head is defined as a linear layer with num_classes output units, where num_classes is the number of possible output classes.
        # For binary sentiment classification, num_classes would be 2, and the classification_head would output a vector of size 2 containing the probabilities of the input text being positive or negative.
        # The predicted class would be the one with the highest probability. I hope this helps clarify the difference between the sentiment prediction and classification tasks, and the role of the classification_head
        # in each task! Let me know if you have any other questions.

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, sentiment = batch
        outputs = self.encoder_decoder(input_ids, attention_mask=attention_mask)
        encoded_text = outputs.last_hidden_state
        sentiment_pred = outputs.pooler_output
        sentiment_loss = self.sentiment_loss_fn(sentiment_pred, sentiment.float())
        self.log('val_sentiment_loss', sentiment_loss)
        return {'val_loss': sentiment_loss}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, sentiment = batch
        outputs = self.encoder_decoder(input_ids, attention_mask=attention_mask)
        encoded_text = outputs.last_hidden_state
        sentiment_pred = outputs.pooler_output
        sentiment_loss = self.sentiment_loss_fn(sentiment_pred, sentiment.float())
        self.log('test_sentiment_loss', sentiment_loss)
        return {'test_loss': sentiment_loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=5e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=1000)
        return [optimizer], [scheduler]


# load the dataset
# dataset = load_dataset("csv", data_files="sentiment_data.csv")

# df = pd.read_csv('sentiment_data.csv')
# df = pd.DataFrame(df)
# dataset = Dataset.from_pandas(df)
# # split the training data into training and validation sets
train_data, val_data = train_test_split(dataset = Dataset.from_pandas(df), test_size=0.2, random_state=42)



imdb = load_dataset("imdb")
print(imdb['train'][0])

EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased', 'bert-base-uncased')
# label2id = {"contradiction": 0, "neutral": 1, "entailment": 2}
# mnli = load_dataset("glue", "mnli", split="train")
# mnli_aligned = mnli.align_labels_with_mapping(label2id, "label")

# # Process the dataset - add a column with the length of the context texts
# dataset_with_length = squad_dataset.map(lambda x: {"length": len(x["context"])})


# extract the training and test splits
train_data = imdb['train']
test_data = imdb['test']
val_data = imdb['unsupervised']

# create the PyTorch Lightning model
model = SentimentModel(num_classes=2, batch_size=16, num_workers=0)

# # Prepare the data
# model.prepare_data()

# define the training parameters
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Callbacks
# save_ckpt_path = '/Users/jordanharris/Code/PycharmProjects/ai-frames/'
checkpoint_callback = ModelCheckpoint(dirpath=None,
                                      save_top_k=-1,
                                      monitor="pretrain_loss",
                                      every_n_train_steps=600,
                                      train_time_interval=None,
                                      auto_insert_metric_name=True,
                                      # save_on_train_epoch_end=True,
                                      save_last=True,
                                      verbose=False)
early_stop_callback = EarlyStopping(monitor="pretrain_loss", patience=3, verbose=True, mode="max", min_delta=.01)

progress_bar = RichProgressBar()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Trainer
# On an M1 Mac, you do not have access to NVIDIA GPUs,
# which are typically used for GPU acceleration in PyTorch.
# Instead, you can use the built-in Apple Silicon GPUs,
# which are referred to as "Apple GPUs" or "Integrated GPUs".
# To use the Apple GPUs for training in PyTorch,
# you need to set the accelerator argument to 'native' instead of 'auto',
# and remove the gpus argument.


wandb_logger = WandbLogger()
trainer = Trainer(accelerator="mps",
                     devices=1,
                     logger=wandb_logger,
                     # gpus=0,
                     max_epochs=8,
                     min_epochs=6,
                     callbacks=[progress_bar,
                                early_stop_callback,
                                checkpoint_callback],
                     # overfit_batches=0.015,
                     # overfit_batches=3,
                     # auto_scale_batch_size="binsearch",
                     enable_progress_bar=True,
                     log_every_n_steps=10,
                     precision=16,
                     amp_backend="native"
                     )


# train the model
trainer.fit(model, train_dataloader, val_dataloader)

# test the model
trainer.test(model)

# save the trained model
torch.save(model.state_dict(), 'sentiment_model.pt')
# One thing to keep in mind is that changes to the data preparation and model architecture can have downstream effects on the rest of the code, particularly on the loss function and optimizer.
# For example, if you change the number of output classes in the model's classification_head layer, you'll need to modify the loss function to take into account the new number of classes. Similarly, if you change the learning rate or other hyperparameters in the optimizer, you'll need to experiment with different values to find the best settings for your specific use case.
# Additionally, changes to the data loading and preprocessing steps can affect the model's training and validation performance. For example, using a larger batch size can speed up training, but it may also lead to overfitting or instability in the loss function. Similarly, using a different tokenizer or text preprocessing method can affect the quality of the model's predictions, as well as the speed of training and inference.

# https://lightning.ai/pages/community/lightning-releases/pytorch-lightning-1-7-release/