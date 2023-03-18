from pt_model_recorder import PtModelRecorder, PtGpt2Recorder
import redis
import time
# redis_client = redis.Redis(host="192.168.53.8", port=6379, db=0, client_name=None)
# recorder = PtModelRecorder(redis_client = redis_client, prefix="LLAMA-7B_0313_")
# # while (recorder.is_recorded() and len(list(recorder.keys('output-output'))) > 0) == False:
# #     print(recorder.is_recorded(), len(list(recorder.keys('output-output'))) > 0 if recorder.is_recorded() else False)
# #     time.sleep(5)
# # # print(recorder.is_recorded(), len(list(recorder.keys('output-output'))) > 0)
# print(recorder.is_recorded())
# print(recorder.exists('output-output'))

token = 'lixiangyu'
print("{}".format(token))

