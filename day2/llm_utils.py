import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 初始化 DeepSeek 客户端
_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)


def first_call_llm(prompt, temperature=0.0, max_tokens=500, retries=3):
    """
    调用大模型 API,自动重试
    
    参数:
        prompt (str): 用户输入
        temperature (float): 0.0~1.5
        max_tokens (int): 最大返回长度
        retries (int): 最大重试次数
    
    返回:
        str: AI 回复文本
    """
    for attempt in range(retries):
        try:
            response = _client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=30  # 30 秒超时
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(f"[重试 {attempt+1}/{retries}] 出错: {e}")
            if attempt < retries - 1:
                time.sleep(2)  # 等 2 秒再试
            else:
                return f"[错误] 重试 {retries} 次后仍然失败: {e}"