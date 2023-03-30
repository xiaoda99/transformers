from sentencepiece import SentencePieceProcessor
from logging import getLogger
from typing import List
import torch
import os
import numpy as np
class LLAMATokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)
#         logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
#         logger.info(
#             f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
#         )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s, bos = True, eos = False, return_tensors = 'False'):
        assert type(s) is str
        t = self.sp_model.Encode(s, add_bos = bos, add_eos = eos)
        return torch.Tensor(t).long().view(1, -1) if return_tensors == 'pt' else t

    def decode(self, t):
        return self.sp_model.Decode(t)
    
    def tokenize(self, s, bos= True, eos = False):
        t = self.sp_model.Encode(s, str, add_bos = bos, add_eos = eos)
        return t
        
    def convert_tokens_to_ids(self, s):

        return self.sp_model.PieceToId(s)
    def convert_ids_to_tokens(self, s):
        # print(type(s))
        if type(s) is np.ndarray or type(s) is torch.Tensor:
            s = s.tolist()
        return self.sp_model.IdToPiece(s)