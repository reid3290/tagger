import requests
import os

sighan_root_dir = "/Users/reid/graduate/data/SIGHAN-bakeoff-2005"
test_file_path = os.path.join(sighan_root_dir, 'testing/pku_test.utf8')
result_file_path = os.path.join(sighan_root_dir, 'ltp-pku-result.txt')
# test_file_path = os.path.join(sighan_root_dir, 'testing/msr_test.utf8')
# result_file_path = os.path.join(sighan_root_dir, 'jieba-msr-result.txt')
result_file = open(result_file_path, 'w')

url = "http://api.ltp-cloud.com/analysis/"
querystring = {"api_key": "r1I7o0E7W6FsxZqONfVJUYTavNZRWkWQpQTeJN8G", "pattern": "ws", "format": "plain"}

for line in open(test_file_path).readlines():
    querystring['text'] = line
    request_successful = False
    response = requests.request("GET", url, params=querystring)
    if response.status_code != 200:
        print response.status_code
        print line
        result = line + "*" * 20
        result_file.write('utf8')
    else:
        result = response.text
        result_file.write(result.encode('utf8'))
