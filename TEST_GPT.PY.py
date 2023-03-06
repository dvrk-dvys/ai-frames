# import os

# from datetime import datetime
# from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.accelerators import CPUAccelerator, MPSAccelerator
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import seaborn as sns

import torch
# from torch import optim, nn, utils, Tensor
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# Huggingface
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DistilBertTokenizerFast,
    DistilBertConfig,
    get_linear_schedule_with_warmup,
    EncoderDecoderModel,
    DistilBertTokenizer,
    BertTokenizerFast,
    AutoTokenizer
)
from datasets import load_dataset


class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['text']
        label = self.data[idx]['label']
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        # label_tensor = torch.tensor(label, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        label_tensor = label_tensor.float()

        # Convert label to tensor
        return {'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'label': label_tensor}


def SentimentCollate_fn(batch):
    """Function to create a mini-batch from a list of samples.
    Args:
        batch: A list of samples from the dataset.
    Returns:
        A mini-batch of tensors.
    """
    # texts, attention_masks, sentiment = zip(*batch)

    texts = [item["input_ids"] for item in batch]
    attention_masks = [item["attention_mask"] for item in batch]
    sentiment = [item["label"] for item in batch]

    padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    # padded_sentiment = pad_sequence(sentiment, batch_first=True, padding_value=0)
    # label_tensor = torch.tensor(sentiment).unsqueeze(1)
    # label_tensor = label_tensor.float()
    batch_size = len(batch)
    padded_label_tensor = torch.nn.functional.pad(sentiment, (0, 0, 0, 128 - batch_size), value=0)

    # padded_classification = torch.stack(classification)
    # return padded_texts, padded_attention_masks, padded_sentiment, padded_classification
    return {
        "input_ids": padded_texts,
        "attention_mask": padded_attention_masks,
        "label": padded_label_tensor,
    }


class SentimentDataModule(LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32, num_workers=0):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

    def setup(self, stage=None):
        # Load and preprocess the dataset
        self.train_dataset = SentimentDataset(self.train_data, self.tokenizer)
        self.val_dataset = SentimentDataset(self.val_data, self.tokenizer)
        self.test_dataset = SentimentDataset(self.test_data, self.tokenizer)
        self.data_collator = SentimentCollate_fn
        # self.data_collator = pad_sequence

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          collate_fn=self.data_collator
                          )

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.data_collator
                          )

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=self.num_workers,
                          collate_fn=self.data_collator
                          )


class SentimentModel(pl.LightningModule):
    def __init__(self, num_classes, batch_size=32, num_workers=0):
        super().__init__()
        self.encoder_decoder = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-uncased',
                                                                                   'bert-base-uncased')
        # self.classification_head = torch.nn.Linear(768, num_classes) # num_classes is the number of output classes
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        # dataset = Dataset.map(lambda examples: self.tokenizer(examples["text"], return_tensors="np"), batched=True)
        self.data_collator = SentimentCollate_fn
        # self.classification_loss_fn = torch.nn.CrossEntropyLoss()
        self.sentiment_loss_fn = torch.nn.MSELoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder_decoder(input_ids, attention_mask=attention_mask, labels=labels)
        encoded_text = outputs.last_hidden_state
        sentiment = outputs.pooler_output
        # classification_pred = self.classification_head(sentiment)
        return {"encoded_text": encoded_text, "sentiment": sentiment}

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, sentiment = batch
        outputs = self.encoder_decoder(input_ids, attention_mask=attention_mask)
        encoded_text = outputs.last_hidden_state
        sentiment_pred = outputs.pooler_output
        sentiment_loss = self.sentiment_loss_fn(sentiment_pred, sentiment.float())
        self.log('train_sentiment_loss', sentiment_loss)
        return sentiment_loss

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


imdb = load_dataset("imdb")
print(imdb['train'][0])

# extract the training and test splits
train_data = imdb['train']
test_data = imdb['test']
val_data = imdb['unsupervised']

data_module = SentimentDataModule(train_data, val_data, test_data, batch_size=16, num_workers=0)

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

trainer = pl.Trainer(accelerator="auto",
                     devices=1,
                     # gpus=0,
                     # logger=wandb_logger,
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
trainer.fit(model, datamodule=data_module)

# test the model
trainer.test(datamodule=data_module, ckpt_path=None)