#%%

import enum
from utils import PAN2020
from xai import ClassificationIntegratedGradients
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Dataset
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dir = "/darkweb_ds/closed_splits/closed_split_v1/xs/pan20-av-small-test.jsonl"
model_root = (
    "/root/vanilla_simpletransformers/Authorship_Models/authorship-chkps-jul2021-large"
)
model_type = "bert-base-uncased-closed-v1-xs-panlm-max642-pan_lmloss0.5234"

model_path = os.path.join(model_root, model_type)

model = BertForSequenceClassification.from_pretrained(
    model_path, output_attentions=True
)
model.to(device)
model.eval()
model.zero_grad()

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

seq_len = 64
special_tokens = 3
test_ds = PAN2020(test_dir, tokenizer, seq_len, special_tokens)
test_dataloader = DataLoader(
    test_ds, sampler=SequentialSampler(test_ds), batch_size=1, num_workers=1
)

xai_model = ClassificationIntegratedGradients(model, model.bert.embeddings, tokenizer)

label_names = ["different", "same"]


#%%
for idx, (x, label) in enumerate(test_dataloader):
    if idx == 10:
        break

    input_ids = x[0]["token_ids"].to(device)
    attn_msk = x[0]["attention_mask"].to(device)
    token_type_ids = x[0]["token_type_ids"].to(device)
    label = label.item()

    _ = xai_model.compute_attribution(
        input_ids=input_ids, attribution_idx=1, true_label=label, label_names=label_names
    )

#%%

# x = next(iter(test_dataloader))
# input_ids = x[0][0]["token_ids"].to(device)
# attn_msk = x[0][0]["attention_mask"].to(device)
# token_type_ids = x[0][0]["token_type_ids"].to(device)
# label = x[1].item()

# label_names = ["different", "same"]
# _ = xai_model.compute_attribution(
#     input_ids=input_ids, attribution_idx=1, true_label=label, label_names=label_names
# )

# %%
