from collections import defaultdict
import torch
import io
import os
import redis
import numpy as np
from shutil import copyfile
from functools import partial
from datetime import datetime
from time import time


def unroll_tensor(inter_result):
    if type(inter_result) == torch.Tensor:
        yield inter_result
    else:
        tensors = []
        for t in inter_result:
            if type(t) == torch.Tensor:
                yield t
            else:
                unroll_tensor(t)

def th_dumps(tensors):
    in_buffer = io.BytesIO()
    torch.save(tensors, in_buffer)
    in_buffer.seek(0)
    return in_buffer.read()

def th_loads(serialized_tensor):
    out_buffer = io.BytesIO()
    out_buffer.write(serialized_tensor)
    out_buffer.seek(0)
    tensor = torch.load(out_buffer)
    return tensor

class PtModelRecorder:
    def __init__(self, prefix="", **kwargs):
        if prefix:
            self.prefix = prefix
        else:
            self.prefix = "RECORD_ON_{}-".format(datetime.now().strftime("%Y%m%d"))

        self.handles = {}
        self._keys = None

    def is_recorded(self):
        raise NotImplementedError

    def _save(self, tensor, key):
        raise NotImplementedError

    def _load(self, key):
        raise NotImplementedError

    def commit(self):
        return True

    def _save_tensor(self, tensor, key):
        tensors = [t.cpu() for t in unroll_tensor(tensor)]
        self._save(tensors, key)

    def get(self, key):
        assert key.endswith("-input") or key.endswith("-output")
        return self._load(key)

    def register_hook(self, hug_model, hook_pattern=lambda x: True):
        def save_activation(name, mod, inp, out):
            self._save_tensor(inp, name + "-input")
            # self._save_tensor(inp, name + "-input")
            self._save_tensor(out, name + "-output")

        for name, layer in hug_model.named_modules():
            if hook_pattern(name):
                self.handles[name] = layer.register_forward_hook(partial(save_activation, name))

        return self.handles

    def clear(self):
        """Clear all recorded information in redis."""
        raise NotImplementedError

    def destroy(self):
        """Remove all hook, and Delete all record"""
        for k, v in handles.items():
            handles[k].remove()

    def keys(self, pattern=""):
        raise NotImplementedError


class RedisRecorder(PtModelRecorder):
    def __init__(self, prefix="", redis_client=None):
        if redis_client:
            self.redis_client = redis_client
        else:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0, client_name=None)

        super(RedisRecorder, self).__init__(prefix=prefix)

    def _save(self, tensors, key):
        buff = th_dumps(tensors)
        set_success = self.redis_client.set(key, buff)
        return set_success

    def _load(self, key):
        key = key.encode()
        assert type(key) == bytes

        serialized_tensor = self.redis_client.get(key)
        tensor = th_loads(serialized_tensor)
        return tensor


class ShareMemRecorder(PtModelRecorder):
    def __init__(self, prefix=""):
        if not os.path.exists("/dev/shm/{}/".format(prefix)):
            os.mkdir("/dev/shm/{}/".format(prefix))

        super(ShareMemRecorder, self).__init__(prefix=prefix)

    def _save(self, tensors, key):
        np_arrays = [t.numpy() for t in tensors]
        with open("/dev/shm/{}/{}.npz".format(self.prefix, key), "wb+") as w:
            try:
                np.savez(w, *np_arrays)
            except Exception as e:
                print("Save failed.", key, np_arrays.shape, np_arrays.dtype)

    def _load(self, key):
        with open("/dev/shm/{}/{}.npz".format(self.prefix, key), "rb") as r:
            np_arrays = np.load(r)
            tensors = [torch.tensor(np_arrays[key]) for key in np_arrays.files]
        return tensors


class BatchShareMemRecorder(PtModelRecorder):
    def __init__(self, prefix=""):
        if not os.path.exists("/dev/shm/{}/".format(prefix)):
            os.mkdir("/dev/shm/{}/".format(prefix))

        self.inter_result = defaultdict(list)

        super(BatchShareMemRecorder, self).__init__(prefix=prefix)

    def _save(self, tensors, key):
        self.inter_result[key].append(tensors)

    def _load(self, key):
        with open("/dev/shm/{}/{}.npz".format(self.prefix, key), "rb") as r:
            np_arrays = np.load(r)
            tensors = [torch.tensor(np_arrays[key]) for key in np_arrays.files]
        return tensors

    def commit(self):
        for key, value in self.inter_result.items():
            # print(key)
            # if key == 'output-output':
            #     print(len(value))
            #     print(len(value[0]), len(value[1]))
            #     print(value[0][0].shape, value[1][0].shape)
            tensors = map(torch.cat, zip(*value))
            np_arrays = [t.numpy() for t in tensors]
            # np_arrays = [t for t in tensors]
            write_path = "/dev/shm/{}/{}.npz".format(self.prefix, key)
            with open(write_path, "wb+") as w:
                try:
                    np.savez(w, *np_arrays)
                    # torch.save(np_arrays, w)
                except Exception as e:
                    pass
                    print("Save failed.", key, np_arrays.shape, np_arrays.dtype)
            if key == 'output-output':
                target_path = "/dev/shm/{}/{}.npz".format(self.prefix, key + '-final')
                copyfile(write_path, target_path)
        self.inter_result.clear()
    
    def clear(self):
        # 清除该pefix下的所有文件
        del_path = "/dev/shm/{}/".format(self.prefix)
        if not os.path.exists(del_path):
            return
        del_list = os.listdir(del_path)
        # print(del_list)
        for f in del_list:
            file_path = os.path.join(del_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)
    def exists(self, key = ""):
        path = "/dev/shm/{}/{}.npz".format(self.prefix, key)
        return os.path.exists(path)


if __name__ == '__main__':
    in_tensor = [torch.ones(3, 3) for _ in range(4)]
    in_tensor2 = [torch.zeros(3, 3) for _ in range(4)]
    # print(torch.cat([torch.ones(3, 3),torch.ones(3, 3)],dim= 0))
    s_client = BatchShareMemRecorder(prefix="LLAMA-7B_0313_")
    # s_client._save(in_tensor, "test_1")
    # s_client._save(in_tensor2, "test_1")
    # s_client.commit()
    aa = s_client._load("output-output")
    print(len(aa))
    print(aa[0].shape)
    # print(aa[1].shape)
    # s_client.clear()

    # output_tensor = s_client._load("test_1")

    # result = [(in_tensor[i] == output_tensor[i]).all() for i in range(4)]
    # assert all(result)

    # r_client = RedisRecorder(prefix="test")
    # r_client._save(in_tensor, "test_1")
    # output_tensor = r_client._load("test_1")

    # result = [(in_tensor[i] == output_tensor[i]).all() for i in range(4)]
    # assert all(result)
