import os
from volcenginesdkarkruntime import Ark
from openai import OpenAI

# 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
# 初始化Ark客户端，从环境变量中读取您的API Key
#print(os.environ.get("ARK_API_KEY"))
"""
def api_call(question, model="ep-20260529103643-rwptp"):
    client = Ark(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key。此为默认方式，您可根据需要进行修改
        api_key=os.environ.get("ARK_API_KEY"),
    )

    response = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        #model="ep-20260526111615-fkrrc", #豆包-Pro
        #model = "ep-20260326144321-wx97c",#豆包-mini
        #model = "ep-20260529103643-rwptp",#deepseek-3.2
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    #{
                    #    "type": "image_url",
                    #    "image_url": {
                #        "url": "https://ark-project.tos-cn-beijing.ivolces.com/images/view.jpeg"
                #    },
                #},
                {"type": "text", "text": question},
                ],
            }
        ],
    
        # 免费开启推理会话应用层加密，访问 https://www.volcengine.com/docs/82379/1389905 了解更多
        extra_headers={'x-is-encrypted': 'true'},
    )
    return response.choices[0]
"""

def api_call(question, model="qwen3.5-flash"):
    client = OpenAI(
        # 若没有配置环境变量，请用百炼API Key将下一行替换为：api_key="sk-xxx",
        api_key= os.getenv('DASHSCOPE_API_KEY'),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model=model,  # 请替换为模型部署成功后的code
        messages=[
            {"role": "user", "content": question},
        ],
        extra_body={"enable_thinking": False},
    )
    return completion.choices[0]