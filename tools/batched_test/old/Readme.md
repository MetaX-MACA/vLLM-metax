#自动化测试脚本model_list_test.sh使用方法介绍
source model_list_test.sh
脚本支持自动识别空闲卡，并发跑多个模型，达到高效测试的目的。
需要注意一下几点：
# 1.用户可以自定义超时时间控制max_wait_time(默认1h)以及空闲卡占用显存的值MEM_THRESHOLD(默认900MB)

# 2.三个文件 model_list_test.sh online.py models.json 应当放在同一目录下

# 3.为了方便查看log，model.json 文件里，模型路径 model 最后不要有斜杠

# 4.若脚本出现异常中断，导致不能正确识别到空闲卡，可以手动清除/tmp/gpu_lock/ 下的文件

# 5.自己看还好，若是需要合作观看建议在source model_list_test.sh 后追加tee，将终端输出到文件，方便其他人查看。