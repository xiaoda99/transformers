import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset

class CHILDDataset(Dataset):
    def __init__(self, input_strs, tokenizer):
        prompt_token = 'Ġ!'; prompt_id = tokenizer._convert_token_to_id(prompt_token)
        bop_str = 'Instruction: '; bop_id = tokenizer.encode(bop_str)[0]  # 'Inst'
        eop_str = '. For example:'; eop_id = tokenizer.encode(eop_str)[2] # 'Ġexample'
        bos_id = tokenizer._convert_token_to_id('Ġ->')
        eos_id = tokenizer._convert_token_to_id('Ċ')

        if tokenizer.pad_token is None: tokenizer.pad_token = '!'
        self.inputs = tokenizer.batch_encode_plus(input_strs, add_special_tokens=False, padding=True, return_tensors='pt')
        input_ids = self.inputs.input_ids
        self.labels = torch.ones_like(input_ids) * (-100)
        for bi in range(input_ids.size(0)):
            bop_idx = (input_ids[bi] == bop_id).nonzero().squeeze(1)
            eop_idx = (input_ids[bi] == eop_id).nonzero().squeeze(1)
            if len(bop_idx) > 0:
                assert len(bop_idx) == 1 and len(eop_idx) == 1
                bop_idx, eop_idx = bop_idx.item(), eop_idx.item()
                input_ids[bi, bop_idx: eop_idx + 2] *= -1  # use prompt embedding for prompt tokens
            
            bos_indices = (input_ids[bi] == bos_id).nonzero().squeeze(1)
            eos_indices = (input_ids[bi] == eos_id).nonzero()[-len(bos_indices):].squeeze(1)
            for i, (bos_i, eos_i) in enumerate(zip(bos_indices.tolist(), eos_indices.tolist())):
                assert eos_i > bos_i + 1
                if i >= 2: self.labels[bi, bos_i + 1: eos_i] = input_ids[bi, bos_i + 1: eos_i]

    def __len__(self):
        return len(self.inputs['input_ids'])

    def __getitem__(self, i):
        return {'input_ids': self.inputs['input_ids'][i],
                'attention_mask': self.inputs['attention_mask'][i],
                'labels': self.labels[i]}

class WrappedEmbedding(nn.Module):
    def __init__(self, 
                wte: nn.Embedding,
                prompt_id: int = None,
                prompt_len: int = 10, 
                random_range: float = 0.5,
                initialize_from_vocab: bool = True):
        super(WrappedEmbedding, self).__init__()
#         self.wte = wte
#         self.prompt_id = prompt_id
#         self.prompt_len = prompt_len
        self.__dict__.update(locals()); del self.self
        if self.prompt_id is not None:
            self.prompt_embedding = nn.parameter.Parameter(
                self.initialize_embedding(random_range, initialize_from_vocab)).to(self.wte.weight.device)
        else:
            self.prompt_embedding = nn.Embedding(self.prompt_len, self.wte.weight.size(1)).to(self.wte.weight.device)
            assert initialize_from_vocab
            self.init_prompt_embedding_()
#             self.prompt_embedding.weight.data = self.initialize_embedding(random_range, initialize_from_vocab)     
            
    def initialize_embedding(self, random_range: float = 0.5, initialize_from_vocab: bool = True):
        if initialize_from_vocab: return self.wte.weight[:self.prompt_len].clone().detach()
        return torch.FloatTensor(self.prompt_len, self.wte.weight.size(1)).uniform_(-random_range, random_range)
    
    def init_prompt_embedding_(self):
        self.prompt_embedding.weight.data[:] = self.wte.weight[:self.prompt_len]
            
    def forward(self, input_ids):
        if self.prompt_id is not None:
            input_embeds = self.wte(input_ids)
            input_embeds[input_ids == self.prompt_id] = self.prompt_embedding.expand(input_embeds.size(0), -1, -1)
        else: # adapted from cpm-2
            prompt_mask = input_ids < 0
            prompt_ids = -input_ids * prompt_mask
            assert torch.all(prompt_ids < self.prompt_len)
            p_embeds = self.prompt_embedding(prompt_ids) * prompt_mask.float().unsqueeze(-1)
            input_ids = input_ids * ~prompt_mask
            w_embeds = self.wte(input_ids) * (~prompt_mask).float().unsqueeze(-1)
            input_embeds = w_embeds + p_embeds
        return input_embeds

# adapted from cpm-2: https://github.com/TsinghuaAI/CPM-2-Finetune/blob/master/utils.py#L133-L164
def get_params_for_prompt_optimization(module: nn.Module):
    params = []
    for t in module.named_modules():
        if "prompt_embedding" in t[0]:
            params.append({'params': [p for p in list(t[1]._parameters.values()) if p is not None]})
    for t in module.named_parameters():
        if "prompt" not in t[0]:
            t[1].requires_grad_(False)    
    return params

def create_optimizer(model, training_args):
    from torch.nn.parallel.distributed import DistributedDataParallel as DDP
    while isinstance(model, (DDP, )): model = model.module
    we.init_prompt_embedding_()
    param_groups = get_params_for_prompt_optimization(model)
    optimizer = AdamW(param_groups, lr=training_args.learning_rate, 
                      betas=(training_args.adam_beta1, training_args.adam_beta2),eps=training_args.adam_epsilon)
    return optimizer

if False:
    wte = model.get_input_embeddings()
    if hasattr(wte, 'wte'): wte = wte.wte  # already been wrapped
    we = WrappedEmbedding(wte, prompt_len=10000)
    model.set_input_embeddings(we)