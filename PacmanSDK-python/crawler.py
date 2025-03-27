import requests
import os
import traceback

BASE_URL = "https://api.saiblo.net/api/matches/"
DOWNLOAD_URL_TEMPLATE = "https://api.saiblo.net/api/matches/{}/download/"
HEADERS = {
    "accept": "application/json, text/plain, */*",
    "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoiYWNjZXNzIiwiZXhwIjoxNzQzNTEwMDAyLCJpYXQiOjE3NDI5MDUyMDIsImp0aSI6IjkwMzViMDAzODI1YjRjZjA4MGZmMWQ4ZDZkNjRhZWU1IiwidXNlcl9pZCI6NjU2Nn0.OgbrWqYcJhdLaanm-Zw2KGKtlQc1RQrJbWu8QzCHy0g",
    "origin": "https://www.saiblo.net",
    "referer": "https://www.saiblo.net/"
}

GAME_ID = 42
LIMIT = 20

FIRST_PLAYERS = {
    "98864c00-31b3-4b42-b244-fc8110618aef",
    "15a594f0-bd1d-4138-808c-223a5d103724",
}

FAIL_OFFSET_LIST = []
FAIL_MATCHID_LIST = []

def log_exception(exc_type, exc_value, exc_traceback):
    with open("crawler_error.log", "a") as f:
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)

def fetch_match_list(offset):
    params = {"limit": LIMIT, "offset": offset, "game": GAME_ID}
    
    try:
        response = requests.get(BASE_URL, params=params, headers=HEADERS)
    except:
        print(f"offset {offset} 获取失败")
        with open("fail_offset.txt", "a") as f:
            print(offset, file=f)
        return []
    
    
    if response.status_code == 200:
        data = response.json()
        return data.get("results", [])
    else:
        with open("fail_offset.txt", "a") as f:
            print(offset, file=f)
        print(f"offset {offset} 请求失败")
        return []

def download_match(match_id, dir):
    url = DOWNLOAD_URL_TEMPLATE.format(match_id)
    
    try:
        response = requests.get(url, headers=HEADERS)
    except:
        print(f"获取match_id {match_id}失败")
        with open("fail_match.txt", "a") as f:
            print(match_id, file=f)
        return None
    
    if response.status_code == 200:
        filename = os.path.join(dir, f"{match_id}.jsonl")
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"已保存 match_id {match_id} 到文件夹 {dir}")
    else:
        print(f"下载match_id {match_id}失败，状态码：{response.status_code}")
        with open("fail_match.txt", "a") as f:
            print(match_id, file=f)
        return None

def main():
    base_dir = "matchdata_json"
    os.makedirs(base_dir, exist_ok=True)

    for offset in reversed(range(30000, 83740, LIMIT)):
        match_list = fetch_match_list(offset)
        
        if not match_list:
            continue
        
        for match in match_list:
            if match.get("state") == "评测成功":
                match_id = match.get("id")
                info = match.get("info", [])
                if not info:
                    continue

                first_player = info[0].get("code", {}).get("id", "unknown")

                if first_player in FIRST_PLAYERS:
                    user_dir = os.path.join(base_dir, first_player)
                    os.makedirs(user_dir, exist_ok=True)

                    download_match(match_id=match_id, dir=user_dir)

if __name__ == "__main__":
    main()