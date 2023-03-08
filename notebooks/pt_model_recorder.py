import torch
import io
import redis
from functools import partial
from datetime import datetime


def unroll_tensor(inter_result):
    tensors = []
    for t in inter_result:
        if type(t) == torch.Tensor:
            yield t
        else:
            unroll_tensor(t)

class PtModelRecorder:
    def __init__(self, redis_client=None, prefix=""):
        if redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, client_name=None)

        if prefix:
            self.prefix = prefix
        else:
            self.prefix = "RECORD_ON_{}-".format(datetime.now().strftime("%Y%m%d"))

        self.handles = {}
        self._keys = None

    def is_recorded(self):
        return len(self.redis_client.keys("{}*".format(self.prefix))) != 0

    def _save_tensor_in_redis(self, tensor, key):
        key = self.prefix + key
        tensors = [t.cpu() for t in unroll_tensor(tensor)]

        in_buffer = io.BytesIO()
        torch.save(tensor, in_buffer)
        in_buffer.seek(0)
        set_success = self.redis_client.set(key, in_buffer.read())
        in_buffer.close()
        return set_success

    def _load_tensor_from_redis(self, key):
        key = key.encode()
        assert type(key) == bytes

        out_buffer = io.BytesIO()
        serialized_tensor = self.redis_client.get(key)
        out_buffer.write(serialized_tensor)
        out_buffer.seek(0)
        tensor = torch.load(out_buffer)
        return tensor

    def get(self, key):
        key = self.prefix + key
        assert key.endswith("-input") or key.endswith("-output")
        return self._load_tensor_from_redis(key)

    def register_hook(self, hug_model):
        def save_activation(name, mod, inp, out):
            self._save_tensor_in_redis(inp, name + "-input")
            self._save_tensor_in_redis(out, name + "-output")

        for name, layer in hug_model.named_modules():
            self.handles[name] = layer.register_forward_hook(partial(save_activation, name))

        return self.handles

    def clear(self):
        """Clear all recorded information in redis."""
        for key in self.redis_client.keys(self.prefix + "*"):
            self.redis_client.delete(key)

    def destroy(self):
        """Remove all hook, and Delete all record"""
        for k, v in handles.items():
            handles[k].remove()

        for key in self.redis_client.keys(self.prefix + "*"):
            self.redis_client.delete(key)

    def keys(self, pattern=""):
        assert self.is_recorded()
        if not self._keys:
            self._keys = [k.decode()[len(self.prefix):] for k in self.redis_client.keys(self.prefix + "*")]
            self._keys.sort()

        return filter(lambda x: pattern in x, self._keys)

class PtGpt2Recorder(PtModelRecorder):
    def get_inputs(self, tokenizer=None):
        "transformer.wte"
        input_tensor = self.get("transformer.wte-input")
        input_toks = []
        if tokenizer:
            input_toks = tokenizer.batch_decode(input_tensor[0])
        return (input_toks, input_tensor)

    def get_output_logits(self):
        "lm_head"
        output_tensor = self.get("lm_head-output")
        return output_tensor

class PtGpt2Recorder(PtModelRecorder):
    def get_inputs(self, tokenizer=None):
        "transformer.wte"
        input_tensor = self.get("transformer.wte-input")
        input_toks = []
        if tokenizer:
            input_toks = tokenizer.batch_decode(input_tensor[0])
        return (input_toks, input_tensor)

    def get_output_logits(self):
        "lm_head"
        output_tensor = self.get("lm_head-output")
        return output_tensor


if __name__ == '__main__':
    from transformers import GPT2Tokenizer, GPT2Model

    tokenizer = GPT2Tokenizer.from_pretrained('/nas2/hyy/pretrain/gpt2-medium/')
    model = GPT2Model.from_pretrained('/nas2/hyy/pretrain/gpt2-medium/')

    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    recorder = PtModelRecorder(prefix="UT-ON-GPTm_")
    recorder.register_hook(model)

    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)

