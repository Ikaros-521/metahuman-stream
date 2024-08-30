import requests
import traceback, json

def send_request(url: str, method='GET', json_data=None, resp_data_type="json", timeout=60):
    """
    发送 HTTP 请求并返回结果

    Parameters:
        url (str): 请求的 URL
        method (str): 请求方法，'GET' 或 'POST'
        json_data (dict): JSON 数据，用于 POST 请求
        resp_data_type (str): 返回数据的类型（json | content）
        timeout (int): 请求超时时间

    Returns:
        dict|str: 包含响应的 JSON数据 | 字符串数据
    """
    headers = {'Content-Type': 'application/json'}

    try:
        if method in ['GET', 'get']:
            response = requests.get(url, headers=headers, timeout=timeout)
        elif method in ['POST', 'post']:
            response = requests.post(url, headers=headers, data=json.dumps(json_data), timeout=timeout)
        else:
            raise ValueError('无效 method. 支持的 methods 为 GET 和 POST.')

        # 检查请求是否成功
        response.raise_for_status()

        if resp_data_type == "json":
            # 解析响应的 JSON 数据
            result = response.json()
        else:
            result = response.content
            # 使用 'utf-8' 编码来解码字节串
            result = result.decode('utf-8')

        return result

    except requests.exceptions.RequestException as e:
        print.error(traceback.format_exc())
        print.error(f"请求出错: {e}")
        return None
        
# 发送音频播放信息给AI Vtuber的http服务端
def send_audio_play_info_to_callback(wait_play_audio_num: int=0):
    """发送音频播放信息给AI Vtuber的http服务端

    Args:
        wait_play_audio_num: 待播放音频数量
    """
    data = {
        "type": "audio_playback_completed",
        "data": {
            # 待播放音频数量
            "wait_play_audio_num": wait_play_audio_num,
        }
    }

    # 请求地址为AI Vtuber API接口，如果你改了配置请自行适配
    resp = send_request('http://127.0.0.1:8082/callback', "POST", data)

    return resp
