import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict
from captum.attr import IntegratedGradients, LayerIntegratedGradients
from captum.attr import visualization as viz

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ClassificationIntegratedGradients:
    def __init__(self, model, ig_layer, tokenizer, device=None):
        self.transformer = model
        self.tokenizer = tokenizer
        self.ref_token_id = self.tokenizer.pad_token_id
        self.lig = self._get_lig(ig_layer)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

    def _custom_forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        attribution_idx=0,
    ):
        """Helper function for Captum's IG."""
        outputs = self.transformer(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )

        preds = outputs[0]
        logits = torch.softmax(preds, dim=1)[:, attribution_idx]
        return logits

    def _construct_ref_input(self, input_ids, ref_token="PAD"):
        if len(input_ids.shape) == 2:
            input_ids = input_ids.squeeze(0)

        if ref_token == "PAD":
            ref_token_id = self.tokenizer.pad_token_id
        elif ref_token == "MASK":
            ref_token_id = self.tokenizer.mask_token_id

        special_tok_ids = [
            self.tokenizer.sep_token_id,
            self.tokenizer.cls_token_id,
            self.tokenizer.pad_token_id,
        ]

        ref_input_ids = [
            tok if tok in special_tok_ids else ref_token_id for tok in input_ids
        ]
        ref_input_ids = torch.tensor(ref_input_ids).unsqueeze(0)
        return ref_input_ids

    def _get_lig(self, ig_layer):
        return LayerIntegratedGradients(self._custom_forward, ig_layer)

    def compute_attribution(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        attribution_idx=0,
        n_steps=50,
        visualize=True,
        true_label=None,
        label_names=None,
    ):
        """To do: token type ids and attention mask"""
        ref_input_ids = self._construct_ref_input(input_ids).to(self.device)
        input_ids = input_ids.to(self.device)
        attributions, convergence_delta = self.lig.attribute(
            inputs=input_ids,
            baselines=ref_input_ids,
            return_convergence_delta=True,
            n_steps=n_steps,
        )

        # Summarize by sum and normalization of the attributions
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)

        if visualize:
            if true_label == None or label_names == None:
                print(
                    f"Expected parameters true_label and label_names to be different than None. Got {true_label}, {label_names}. Only the attributions will be returned."
                )

            model_output = self.transformer(input_ids)
            pred_idx = torch.argmax(model_output[0]).item()
            pred_scores = torch.softmax(model_output[0], dim=1)[0]
            score = pred_scores[pred_idx].item()

            tokens_list = input_ids[0].detach().tolist()
            tokens = self.tokenizer.convert_ids_to_tokens(tokens_list)
            position_vis = viz.VisualizationDataRecord(
                attributions,
                score,
                label_names[pred_idx],
                label_names[true_label],
                label_names[attribution_idx],
                attributions.sum(),
                tokens,
                convergence_delta,
            )

            _ = viz.visualize_text([position_vis])

        return attributions
