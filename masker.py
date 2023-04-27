import torch

import random
from typing import List, Tuple
import spacy
from dateparser.search import search_dates
import re

# first, we have to download the norwegian model with `python3 -m spacy download nb_core_news_sm`
# we also have to pip-install dateparser


class Masker:
    def __init__(self, tokenizer, mask_p=0.15, random_factor=1.0, salient_factor=1.0, n_special_tokens=105, seq_length=128):
        self.tokenizer = tokenizer
        self.n_special_tokens = n_special_tokens
        self.mask_p = mask_p
        self.seq_length = seq_length
        self.random_factor = random_factor
        self.salient_factor = salient_factor
        self.ner_model = spacy.load(
            "nb_core_news_sm",  # smallest & dumbest CPU model
            disable=[
                "tok2vec",
                "morphologizer",
                "parser",
                "lemmatizer",
                "attribute_ruler"
            ]
        )
        self.mask_id = tokenizer.convert_tokens_to_ids("[MASK]")
    
    def __call__(self, input_ids: torch.Tensor, raw_text: str, offsets: List[List[int]]):
        random_mask = self.get_random_mask(input_ids, self.mask_p * self.random_factor)
        if self.salient_factor:
            salient_mask = self.get_salient_mask(raw_text, offsets)
            # random_mask = self.get_random_mask(input_ids, self.mask_p * self.random_factor)

            masked_ids = torch.where(salient_mask | random_mask, self.mask_id, input_ids)
        else:
            masked_ids = torch.where(random_mask, self.mask_id, input_ids)
        return masked_ids

    def get_salient_mask(self, raw_text: str, offsets: List[Tuple[int]]):


        entities_dict = self.ner_model(raw_text).ents
        entities = []
        for entity in entities_dict:
            if not (entity.text == "UNK"):
                entities.append((entity.start_char, entity.end_char))

        try:
            dates = search_dates(raw_text, languages=["nb", "nn"])
            dates = [
                (m.start(), m.end())
                for date, _ in dates
                for m in re.finditer(date, raw_text)
            ]
            spans = entities + dates
        except:
            spans = entities

        if spans: #If there is entities or dates found
            n_masked = torch.binomial(
                torch.tensor([len(spans)], dtype=torch.float),
                torch.tensor([self.mask_p])
            ).long().item()
            n_masked = max(1, n_masked)
            masked_spans = random.sample(spans, k=n_masked)
        else:
            masked_spans = []

        mask = [
            any(start >= a and end <= b and end > 0 for a, b in masked_spans)
            for start, end in offsets
        ]

        mask += (self.seq_length - len(mask)) * [False] #for the non-padded offsets for the remainder chunks
        
        mask = torch.tensor(mask)
        return mask

    def get_random_mask(self, input_ids: torch.Tensor, mask_p: float):
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        preservation_mask = input_ids < self.n_special_tokens

        n_masked = torch.binomial(
            (input_ids >= self.n_special_tokens).float().sum(0, keepdim=True),
            torch.FloatTensor([mask_p])
        ).item()

        counter = 100  # safety first!
        while mask.sum() <= n_masked and counter > 0:
            span_length = torch.tensor([0]).geometric_(1/3).item() % 10
            offset = torch.randint(-(span_length - 1), input_ids.size(0) + span_length, []).item()
            mask[max(0, offset) : min(mask.size(0)-1, offset + span_length)] = True
            mask[preservation_mask] = False

            counter -= 1

        return mask


if __name__ == "__main__":

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("ltgoslo/norbert2")

    text = """Sporvejsmuseet Skjoldenæsholms historie går fra stiftelsen 1965 over indvielsen af museet 1978 og frem til i dag. Sporvejshistorisk Selskab blev stiftet på et tidspunkt, hvor sporvognene i København var ved at blive erstattet af busser. Man besluttede derfor at bevare nogle af de gamle sporvogne, mens tid var."""
    #text = """Badminton er en sport som utøves av to lag med enten eller to spillere på en bane delt av et nett. I motsetning til i andre racketspill, benytter man ikke ball i badminton. Derimot spilles det med et prosjektil lagd av fjær eller en variasjon av plast, som gjerne kalles flue, dusk eller ball. Norges Badminton Forbund ble stiftet i 1938. Norgesmesterskapet i badminton har blitt arrangert siden 1939. Kongepokalen i badminton har blitt delt ut siden 1949. Historie. Badminton er en gammel idrett med over tusen års tradisjon i Kina, og Thailand. """
    
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )

    # TEST 1

    masker = Masker(tokenizer, mask_p=1.0, random_factor=0.0)
    masked_ids = masker(torch.tensor(encoding.input_ids), text, encoding.offset_mapping)

    for pre, post in zip(encoding.input_ids, masked_ids.tolist()):
        print(f"{tokenizer.convert_ids_to_tokens(pre)} -> {tokenizer.convert_ids_to_tokens(post)}")

    print()
    print("___________")
    print()

    # TEST 2

    masker = Masker(tokenizer, mask_p=0.15, random_factor=1.0)
    masked_ids = masker(torch.tensor(encoding.input_ids), text, encoding.offset_mapping)

    for pre, post in zip(encoding.input_ids, masked_ids.tolist()):
        print(f"{tokenizer.convert_ids_to_tokens(pre)} -> {tokenizer.convert_ids_to_tokens(post)}")
