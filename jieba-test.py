import jieba
import os
sighan_root_dir = "/Users/reid/graduate/data/SIGHAN-bakeoff-2005"
# test_file_path = os.path.join(sighan_root_dir, 'testing/pku_test.utf8')
# result_file_path = os.path.join(sighan_root_dir, 'jieba-pku-result.txt')
test_file_path = os.path.join(sighan_root_dir, 'testing/msr_test.utf8')
result_file_path = os.path.join(sighan_root_dir, 'jieba-msr-result.txt')
result_file = open(result_file_path, 'w')
for line in open(test_file_path).readlines():
    result = " ".join(jieba.cut(line))
    result_file.write(result.encode('utf8'))
