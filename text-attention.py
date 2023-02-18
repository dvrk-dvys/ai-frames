import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import EncoderDecoderModel, BertTokenizer
import pytorch_lightning as pl

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

class MyDataModule(pl.LightningDataModule):
    def __init__(self, train_data, val_data, test_data, batch_size=32):
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = MyDataset(self.train_data)
        self.val_dataset = MyDataset(self.val_data)
        self.test_dataset = MyDataset(self.test_data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)

class MyEncoderDecoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained('bert-base-cased', 'bert-base-cased')

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask).logits

    def training_step(self, batch, batch_idx):
        input_text = batch
        target_text = batch
        input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, return_tensors='pt')['input_ids']
        target_ids = self.tokenizer.batch_encode_plus(target_text, padding=True, return_tensors='pt')['input_ids']
        decoder_input_ids = target_ids[:, :-1]
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        lm_labels = target_ids[:, 1:]
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100
        outputs = self(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), decoder_input_ids=decoder_input_ids, decoder_attention_mask=decoder_attention_mask)
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), lm_labels.view(-1), ignore_index=-100)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        input_text = batch
        target_text = batch
        input_ids = self.tokenizer.batch_encode_plus(input_text, padding=True, return_tensors='pt')['input_ids']
        target_ids = self.tokenizer.batch_encode_plus(target_text, padding=True, return_tensors='pt')['input_ids']
        decoder_input_ids = target_ids[:, :-1]
        decoder_attention_mask = torch.ones_like(decoder_input_ids)
        lm_labels = target_ids[:, 1:]
        lm_labels[lm_labels == self.tokenizer.pad_token_id] = -100
        outputs = self(input_ids=input_ids, attention_mask=input_ids.ne(self.tokenizer.pad_token))
