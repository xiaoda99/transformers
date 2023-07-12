from posixpath import split
import time
import openai
from cachier import cachier
from log import logger
import random
stack = []
# 注意千万不能在这里测试使用key直接测试，导致封号！！！！
with open("/nas/xd/projects/openai_keys.txt", "r") as f:
    for line in f.readlines():
        stack.append(line.strip().split()[0])

# @lru_cache(maxsize=1024)
# @cachier(cache_dir='/nas/xd/.cachier')  # portalocker lockException: [Errno 9] Bad file descriptor, maybe due to nfs
@cachier()  # persistent cache
def query_openai(prompt, engine):

    max_tokens = 20
    # if not hasattr(query_openai, 'cache'): query_openai.cache = {}
    # cache = query_openai.cache; key = (engine, prompt)
    if True: #key not in cache:
        openai.api_key = stack[0]
        proxy_key = "brd-customer-hl_c1b0ccff-zone-openai2-ip-178.171.0.57:sf23ma3ozhu3@zproxy.lum-superproxy.io:22225"
        openai.proxy = {"http": 'http://'+proxy_key, "https": 'https://' + proxy_key}
        response = openai.Completion.create(engine=engine, prompt=prompt,
            max_tokens=max_tokens, temperature=0, echo=False, stop='\n')
        text = response.choices[0].text
        # cache[key] = text
        last_line = prompt.split('\n')[-1]
        # print(f"In query_openai: {last_line} -> {text}")#, cache.size = {len(cache)}")
        logger.info(f"In query_openai: {last_line} -> {text}")
        time.sleep(1.5)
        # time.sleep(max(0, 1.5 - (time.time() - t0)))  # to avoid reaching rate limit of 60 / min
        return text #cache[key]
        


# 两个函数可以合并
# def chatgpt_openai(prompt, retry_count = 0):
    
#     # openai.api_key = ""
#     try:
#         openai.api_key = random.choice(stack)
#         response = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo",
#         messages=[
#                 {"role": "system", "content": "你是一个有帮助的助手，负责检查语法是否有错误。"},
#                 {"role": "user", "content": prompt}
#             ],
#         temperature = 0,
#         )
#         time.sleep(5)
#     except Exception as e:
#         if retry_count <= 3: # 重试4次
#             time.sleep(2 ** (retry_count + 1)) # 指数衰减
#             print("prompt = {},出现异常:{}, [OPEN_AI] RateLimit exceed, 第{}次重试".format(prompt, e, retry_count+1))
#             return chatgpt_openai(prompt, retry_count+1)
#         else:
#             print("prompt = {}, 生成失败, [OPEN_AI] RateLimit exceed, 重试4次失败".format(prompt))
#             return  "Generate False"
#     return response['choices'][0]['message']['content']

if __name__ == "__main__":
    # texts = []
    # with open('/nas/xd/projects/transformers/notebooks/lxy/test.txt', 'r') as f:
    #     texts = f.read().split('====')
    # for text in texts:
    #     # chatgpt_openai(texts[0] + '上述话是否有语法错误？')
    #     print(text + '上述话是否有语法错误？')
    T1 = time.time()
#     text = '''The apple is here.
# The coffee is here.
# The shoes are here.
# The red is here.
# The blueberries are here.
# The blueberries'''
    
    
    print(query_openai("7897dd98",'text-davinci-003'))
    T2 = time.time()
    print(T2 - T1)

        
