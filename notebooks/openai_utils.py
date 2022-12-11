import time
import openai

openai.api_key = open('/nas/xd/projects/openai_api_keys.txt').readlines()[4].split()[0]

# @lru_cache(maxsize=1024)
# @cachier()
# decorators do not work well with autoreload, so implement my own cache instead
def query_openai(prompt, max_tokens=20, engine='text-davinci-002'):
    if not hasattr(query_openai, 'cache'): query_openai.cache = {}
    cache = query_openai.cache; key = (engine, prompt)
    if key not in cache:
        t0 = time.time()
        response = openai.Completion.create(engine=engine, prompt=prompt,
            max_tokens=max_tokens, temperature=0, echo=False, stop='\n')
        text = response.choices[0].text
        cache[key] = text
        last_line = prompt.split('\n')[-1]
        print(f"In query_openai: {last_line} -> {text}, cache.size = {len(cache)}")
        time.sleep(max(0, 1.5 - (time.time() - t0)))  # to avoid reaching rate limit of 60 / min
    return cache[key]