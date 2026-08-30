import time
from concurrent.futures import ThreadPoolExecutor
import judge
t0 = time.time()
with ThreadPoolExecutor(3) as ex:
    list(ex.map(judge.views, judge.S["sample"]))
print("judge views done", round(time.time() - t0))
