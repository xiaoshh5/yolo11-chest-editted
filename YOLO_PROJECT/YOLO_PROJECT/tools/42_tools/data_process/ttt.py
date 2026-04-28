import time

# 获取当前时间戳
current_timestamp = time.time()
print(int(current_timestamp))

# 格式化时间戳为本地时间
local_time = time.localtime(current_timestamp)
print(local_time)

# 获取格式化的时间
formatted_time = time.strftime('%Y-%m-%d %H:%M:%S', local_time)