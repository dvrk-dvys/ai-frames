import torch
from transformers import BertForMaskedLM, BertTokenizer, pipeline
# from transformers import AutoTokenizer, AutoModel
from rich.progress import track
import logging
import matplotlib.pyplot as plt
import torch.nn.functional as F
import re

# https://huggingface.co/Rostlab/prot_bert_bfd
# https://www.kaggle.com/code/ratthachat/proteinbert-lightning-multitasks

# Load the ProtBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, use_fast=True )
model = BertForMaskedLM.from_pretrained("Rostlab/prot_bert_bfd")
unmasker = pipeline('fill-mask', model=model, tokenizer=tokenizer)

vocabulary = {'K': 12, 'D': 14, 'N': 17, 'B': 27, 'C': 23, 'M': 21, 'S': 10, 'G': 7, 'I': 11, 'L': 5, 'A': 6, 'T': 15, 'H': 22, 'Y': 20, 'Q': 18, 'X': 25, 'Z': 28, 'O': 29, 'E': 9, 'W': 24, 'F': 19, 'R': 13, 'U': 26, 'V': 8, 'P': 16}
aa_dict = {
    "K": "Lysine (Lys)",
    "D": "Aspartic acid (Asp)",
    "N": "Asparagine (Asn)",
    "B": "Aspartic acid or Asparagine (Asp or Asn)",
    "C": "Cysteine (Cys)",
    "M": "Methionine (Met)",
    "S": "Serine (Ser)",
    "G": "Glycine (Gly)",
    "I": "Isoleucine (Ile)",
    "L": "Leucine (Leu)",
    "A": "Alanine (Ala)",
    "T": "Threonine (Thr)",
    "H": "Histidine (His)",
    "Y": "Tyrosine (Tyr)",
    "Q": "Glutamine (Gln)",
    "X": "Unknown or ambiguous amino acid",
    "Z": "Glutamic acid or Glutamine (Glu or Gln)",
    "O": "Pyrrolysine (Pyl)",
    "E": "Glutamic acid (Glu)",
    "W": "Tryptophan (Trp)",
    "F": "Phenylalanine (Phe)",
    "R": "Arginine (Arg)",
    "U": "Selenocysteine (Sec)",
    "V": "Valine (Val)",
    "P": "Proline (Pro)"
}



sequence_Example1 = "A E T C Z A O"
masked_sequence = 'D L I P T S S K L V V [MASK] D T S L Q V K K A F F A L V T'
masked_index = masked_sequence.index("[MASK]")

# _______________________Logits_______________________________________
sequence_Example1_ = re.sub(r"[UZOB]", "X", sequence_Example1)
masked_sequence_ = re.sub(r"[UZOB]", "X", masked_sequence)

encoded_input1 = tokenizer(sequence_Example1_, return_tensors='pt')
encoded_masked = tokenizer(masked_sequence_, return_tensors='pt')

output1 = model(**encoded_input1)
output2 = model(**encoded_masked)

# output1.logits[0, masked_index].topk(k=5).indices

pred = unmasker(masked_sequence)

predicted_tokens = []
for a in pred:
    predicted_tokens.append(a['token_str'])

# Convert the token IDs to strings using the tokenizer
# predicted_tokens = tokenizer.convert_ids_to_tokens(pred)

amino_acids = [vocabulary.get(token, "[UNK]") for token in predicted_tokens]

print("Top 5 predicted amino acids for the unknown token:")
print(amino_acids)
for i in predicted_tokens:
    print(aa_dict[i])

# Get the logits for the masked positions in the sequence
logits = output2.logits[0, masked_index]

# Convert logits to probabilities using softmax
probabilities = F.softmax(logits, dim=-1)

# Convert the probabilities tensor to a list
probabilities = probabilities.tolist()

# Get the index of the predicted amino acid for each masked position
predicted_indices = [amino_acids.index(aa) for aa in predicted_tokens]

# Create a list of dictionaries, where each dictionary contains the probability of each amino acid at a masked position
prob_dict_list = []
for i, index in enumerate(predicted_indices):
    prob_dict = {}
    for aa, prob in zip(vocabulary.keys(), probabilities[i]):
        prob_dict[aa] = prob
    prob_dict_list.append(prob_dict)

# Print the probabilities for the first masked position
print("Probabilities for the first masked position:")
print(prob_dict_list[0])



# # Get the attention values for the last layer
# last_layer_attentions = output1.attentions[-1]
#
# # Select the attention values for a specific head and token
# head = 0
# token = 5
# attention_weights = last_layer_attentions[0, head, token]
#
# # Plot the attention weights as a heatmap
# plt.imshow(attention_weights, cmap="hot", interpolation="nearest")
# plt.show()

# _______________________Logits_______________________________________


# # Set up logging and progress bar
# logging.basicConfig(level=logging.INFO)
# # progress = RichProgressBar()
#
# # Example protein sequence
# sequence = "MKKIVLALSLVLVSSVLMAAQGKARGRGGGKGGSARMS"
# masked_sequence = "MKKIVLALSLVLVSSVLMAAQGKARGRGGGKGGSAR[MASK]S"
# masked_index = masked_sequence.index("[MASK]")
#
# encoded_sequence = tokenizer.encode(sequence, add_special_tokens=True, truncation=True, padding=True, max_length=512, return_tensors="pt")
# encoded_masked_sequence = tokenizer.encode(masked_sequence, add_special_tokens=True, truncation=True, padding=True, max_length=512, return_tensors="pt")
#
# # Tokenize the protein sequence
# inputs = tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True)
#
# # Pass the inputs through the ProtBERT model to get the embeddings
# with torch.no_grad():
#     outputs_nomask = model(**inputs, output_hidden_states=True)
#     prediction_ids = model(encoded_masked_sequence).logits[0, masked_index].topk(k=5).indices
#
# embeddings = outputs_nomask.hidden_states[-1][:, 0, :]
#
#
# # Convert the token IDs to strings using the tokenizer
# prediction_tokens = tokenizer.convert_ids_to_tokens(prediction_ids)
# predicted_token_ids = torch.argsort(prediction_tokens, descending=True)[:5]
# predicted_tokens = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in predicted_token_ids]
#
# # predictions = prediction_ids.logits[0, masked_index]
#
#
# # The MaskedLMOutput object contains several attributes,
# # including logits, hidden_states, and attentions,
# # that are produced by the internal layers of the
# # model during the language modeling task.
# # The logits attribute represents the predicted likelihood
# # of each token in the vocabulary at the masked position,
# # and is typically used to generate a list of candidate
# # tokens that could fill the mask.
#
# predicted_token_ids = torch.argsort(prediction_tokens, descending=True)[:5]
# predicted_tokens = [tokenizer.convert_ids_to_tokens([token_id])[0] for token_id in predicted_token_ids]
#
#
#
# # Print the embeddings
# print(embeddings)
#
# seq_len = len(sequence)
#
# vocabulary = {'K': 12, 'D': 14, 'N': 17, 'B': 27, 'C': 23, 'M': 21, 'S': 10, 'G': 7, 'I': 11, 'L': 5, 'A': 6, 'T': 15, 'H': 22, 'Y': 20, 'Q': 18, 'X': 25, 'Z': 28, 'O': 29, 'E': 9, 'W': 24, 'F': 19, 'R': 13, 'U': 26, 'V': 8, 'P': 16}
# amino_acids = [vocabulary.get(token, "[UNK]") for token in predicted_tokens]
# aa_dict = {
#     "K": "Lysine (Lys)",
#     "D": "Aspartic acid (Asp)",
#     "N": "Asparagine (Asn)",
#     "B": "Aspartic acid or Asparagine (Asp or Asn)",
#     "C": "Cysteine (Cys)",
#     "M": "Methionine (Met)",
#     "S": "Serine (Ser)",
#     "G": "Glycine (Gly)",
#     "I": "Isoleucine (Ile)",
#     "L": "Leucine (Leu)",
#     "A": "Alanine (Ala)",
#     "T": "Threonine (Thr)",
#     "H": "Histidine (His)",
#     "Y": "Tyrosine (Tyr)",
#     "Q": "Glutamine (Gln)",
#     "X": "Unknown or ambiguous amino acid",
#     "Z": "Glutamic acid or Glutamine (Glu or Gln)",
#     "O": "Pyrrolysine (Pyl)",
#     "E": "Glutamic acid (Glu)",
#     "W": "Tryptophan (Trp)",
#     "F": "Phenylalanine (Phe)",
#     "R": "Arginine (Arg)",
#     "U": "Selenocysteine (Sec)",
#     "V": "Valine (Val)",
#     "P": "Proline (Pro)"
# }
#
#
#
# print("Top 5 predicted amino acids:")
# print(amino_acids)
# print()
#
#
#
