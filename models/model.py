from openai import OpenAI
from models.base_model import BaseModel
import os
class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        # 明确指定 OpenAI 的官方 API 端点，避免代理或本地干扰
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1"  # 确保使用官方端点
        )

    def generate_response(self, prompt: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content
        except Exception as e:
            # 更详细的错误处理
            if "429" in str(e):
                print("OpenAI API Error: 配额不足，请检查您的账户余额或升级套餐。详情见：https://platform.openai.com/docs/guides/error-codes/api-errors")
            elif "requests.exceptions.SSLError" in str(e):
                print("OpenAI API Error: SSL证书验证失败，请检查网络代理设置。")
            else:
                print(f"OpenAI API Error: {e}")
            return ""


# deepseek
import requests
import json
import time
from typing import Optional, Dict, Generator, Union
from models.base_model import BaseModel
class DeepSeekModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "DeepSeek-R1"):
        super().__init__(api_key, model_name)
        self.base_url = "https://maas-cn-southwest-2.modelarts-maas.com/v1/infers/952e4f88-ef93-4398-ae8d-af37f63f0d8e/v1/chat/completions"
        self.default_timeout = 480

    def generate_response(
            self,
            prompt: str,
            system_prompt: str = "",
            max_tokens: int = 3072,
            temperature: float = 0.3,
            timeout: Optional[int] = None,  # 可自定义超时时间
            **kwargs
    ) -> str:
        """
        生成响应（内部使用流式请求以加快响应速度，但返回完整响应）

        参数:
            prompt: 用户输入的提示词
            system_prompt: 系统角色提示词
            max_tokens: 最大生成token数
            temperature: 采样温度
            timeout: 请求超时时间（秒），默认使用self.default_timeout
            kwargs: 其他API参数
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,  # 始终使用流式请求
            **kwargs
        }

        try:
            # 使用流式请求
            response = requests.post(
                self.base_url,
                headers=headers,
                data=json.dumps(payload),
                verify=False,
                timeout=timeout or self.default_timeout,
                stream=True
            )

            if response.status_code != 200:
                raise Exception(f"API Error [{response.status_code}]: {response.text}")

            # 收集完整响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = line[6:]  # 移除 'data: ' 前缀
                        if data == '[DONE]':
                            break
                        try:
                            json_data = json.loads(data)
                            if 'choices' in json_data and len(json_data['choices']) > 0:
                                delta = json_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    full_response += delta['content']
                        except json.JSONDecodeError:
                            continue

            return full_response

        except requests.exceptions.Timeout:
            print(f"DeepSeek API 请求超时: 请求超过 {timeout or self.default_timeout} 秒未完成")
            return "请求超时，请稍后重试。"
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API 请求失败: {str(e)}")
            return ""
        except Exception as e:
            print(f"DeepSeek API 错误: {str(e)}")
            return ""


# gemini
import requests
import json
from typing import Optional, Dict
from models.base_model import BaseModel
class GeminiModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        super().__init__(api_key, model_name)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"

    def generate_response(
            self,
            prompt: str,
            temperature: Optional[float] = 0.3,
            max_output_tokens: Optional[int] = 1024,
            safety_settings: Optional[Dict] = None,
            **kwargs
    ) -> str:
        """
        生成响应（符合最新API规范）

        参数:
            temperature: 控制采样随机性 (0.0-1.0)
            max_output_tokens: 最大输出token数
            safety_settings: 安全过滤配置
            kwargs: 其他API参数
        """
        headers = {
            "Content-Type": "application/json"
        }

        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {}
        }

        # 添加可选参数
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        if max_output_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
        if safety_settings:
            payload["safetySettings"] = safety_settings
        if kwargs:
            payload.update(kwargs)

        try:
            response = requests.post(
                url=self.base_url,
                params={"key": self.api_key},
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code != 200:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                raise Exception(f"API Error [{response.status_code}]: {error_msg}")

            response_data = response.json()

            # 解析响应结构
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            return ""

        except requests.exceptions.RequestException as e:
            print(f"Gemini API Request Failed: {str(e)}")
            return ""
        except json.JSONDecodeError:
            print(f"Invalid API Response: {response.text}")
            return ""
        except Exception as e:
            print(f"Gemini API Error: {str(e)}")
            return ""


# kimi
import os
from openai import OpenAI
from models.base_model import BaseModel
from typing import List, Dict, Any
class KimiModel(BaseModel):
    # def __init__(self, api_key: str = None, model_name: str = "moonshot-v1-8k"):
    def __init__(self, api_key: str = None, model_name: str = "moonshot-v1-32k"):
        # 如果未提供 api_key，则从环境变量中获取
        api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("API Key 未提供，请设置 MOONSHOT_API_KEY 环境变量或传入 api_key 参数")

        super().__init__(api_key, model_name)
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1"
        )
        # 初始化历史对话，默认包含系统消息
        self.history = [
            {
                "role": "system",
                "content": "你好"
            }
        ]

    def generate_response(self, prompt: str, temperature: float = 0.3, reset_history: bool = False) -> str:
        """
        生成响应，支持多轮对话

        参数:
            prompt: 用户输入的提示词
            temperature: 采样温度，默认 0.3
            reset_history: 是否重置对话历史
        """
        try:
            # 如果重置历史，则恢复到初始状态
            if reset_history:
                self.history = self.history[:1]  # 保留系统消息

            # 添加用户输入到历史
            self.history.append({"role": "user", "content": prompt})

            # 调用 API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                temperature=temperature
            )
            result = response.choices[0].message.content

            # 将助手回复添加到历史
            self.history.append({"role": "assistant", "content": result})

            return result
        except Exception as e:
            print(f"Kimi API Error: {e}")
            return ""

    def get_history(self) -> List[Dict[str, str]]:
        """返回当前对话历史"""
        return self.history

    def clear_history(self):
        """清空对话历史，保留系统消息"""
        self.history = self.history[:1]


# 智谱清言
import requests
from models.base_model import BaseModel
class ZhipuModel(BaseModel):
    # def __init__(self, api_key: str, model_name: str = "glm-4v-flash"):
    def __init__(self, api_key: str, model_name: str = "glm-4-flash"):
        """
        初始化 ZhipuModel 类。

        Args:
            api_key (str): 用户的 API Key
            model_name (str): 模型名称，默认为 "glm-4-plus"（根据官方示例更新）
        """
        super().__init__(api_key, model_name)
        # 根据官方文档推测的正确端点，建议参考最新官方 API 文档确认
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def generate_response(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """
        调用智谱AI API 生成响应。

        Args:
            prompt (str): 用户输入的提示词
            temperature (float): 控制生成文本的随机性，默认 0.7
            max_tokens (int): 最大生成 token 数，默认 1024

        Returns:
            str: API 返回的生成的文本内容，或空字符串（出错时）
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # 支持多轮对话格式，即使当前仅单条消息
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            # 检查 HTTP 状态码
            response.raise_for_status()

            # 解析 JSON 响应
            data = response.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError("API response missing 'choices' field")

            # 返回生成的文本内容
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            # HTTP 请求相关错误
            print(
                f"Zhipu API HTTP Error: {e}, Status Code: {response.status_code if 'response' in locals() else 'N/A'}")
            return ""
        except (KeyError, ValueError) as e:
            # 响应解析错误
            print(f"Zhipu API Response Parsing Error: {e}")
            return ""
        except Exception as e:
            # 其他未预期的错误
            print(f"Zhipu API Unexpected Error: {e}")
            return ""


# 豆包
import requests
from models.base_model import BaseModel
class DoubaoModel(BaseModel):
    # def __init__(self, api_key: str, model_name: str = "doubao-1-5-lite-32k-250115"):
    def __init__(self, api_key: str, model_name: str = "doubao-1-5-pro-32k-250115"):
        super().__init__(api_key, model_name)
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"

    def generate_response(self, prompt: str, temperature: Optional[float] = 0.3, max_tokens: Optional[int] = 1024) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}]
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            # 根据常见API响应结构假设的解析方式，可能需要根据实际响应调整
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"豆包API错误: {e}")
            return ""


# 讯飞星火
import requests
from models.base_model import BaseModel
from typing import Optional, Dict, List, Any
class XunfeiModel(BaseModel):
    # def __init__(self, api_key: str, model_name: str = "lite"):
    def __init__(self, api_key: str, model_name: str = "4.0Ultra"):
        super().__init__(api_key, model_name)
        self.base_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"

    def generate_response(
            self,
            prompt: str,
            temperature: Optional[float] = 0.3,
            max_tokens: Optional[int] = 1024,
            stream: bool = False,
            system_content: str = "你是知识渊博的助理",
            **kwargs: Any
    ) -> str:
        """
        生成响应

        参数:
            prompt: 用户输入的提示词
            temperature: 采样温度 (0-1)
            max_tokens: 最大生成token数
            stream: 是否启用流式输出
            system_content: 系统消息内容
            kwargs: 其他可选参数（如 top_k, presence_penalty 等）
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        # 构造消息列表
        messages = [
            {"role": "user", "content": prompt}
        ]

        # 构造请求体
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs  # 允许传入其他可选参数
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()  # 检查HTTP状态码
            data = response.json()
            # 假设响应结构与OpenAI类似
            return data["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"讯飞星火API错误: {str(e)}")
            return ""


# 腾讯混元
import os
from openai import OpenAI
from models.base_model import BaseModel
from typing import Optional, Dict, Any
class HunyuanModel(BaseModel):
    # def __init__(self, api_key: str = None, model_name: str = "hunyuan-lite"):
    def __init__(self, api_key: str = None, model_name: str = "hunyuan-turbo"):
        # 如果未提供 api_key，则从环境变量中获取
        api_key = api_key or os.environ.get("HUNYUAN_KEY")
        if not api_key:
            raise ValueError("API Key 未提供，请设置 HUNYUAN_API_KEY 环境变量或传入 api_key 参数")

        super().__init__(api_key, model_name)
        # 构造 OpenAI 兼容的 client，指定混元 endpoint
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.hunyuan.cloud.tencent.com/v1"
        )

    def generate_response(
            self,
            prompt: str,
            enable_enhancement: bool = True,
            temperature: Optional[float] = 0.3,
            max_tokens: Optional[int] = 2048,
            **kwargs: Any
    ) -> str:
        """
        生成响应

        参数:
            prompt: 用户输入的提示词
            enable_enhancement: 是否启用增强功能（混元自定义参数）
            kwargs: 其他可选参数
        """
        try:
            # 调用混元 API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                extra_body={"enable_enhancement": enable_enhancement},  # 混元自定义参数
                **kwargs  # 支持其他参数传入
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"腾讯混元 API 错误: {e}")
            return ""


# 文心一言
import requests
from models.base_model import BaseModel
from typing import Optional
class WenxinModel(BaseModel):
    # def __init__(self, api_key: str, secret_key: str, model_name: str = "ernie-4.0-turbo-8k"):
    def __init__(self, api_key: str, secret_key: str, model_name: str = "ernie-4.0-turbo-8k-latest"):
        super().__init__(api_key, model_name)
        self.secret_key = secret_key
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.chat_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_name}"
        self.access_token = self._get_access_token()

    def _get_access_token(self) -> str:
        """获取 access_token"""
        params = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "client_secret": self.secret_key
        }
        try:
            response = requests.get(self.token_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data["access_token"]
        except Exception as e:
            print(f"获取文心一言 access_token 失败: {e}")
            raise

    def generate_response(self, prompt: str, temperature: Optional[float] = 0.3,
            max_tokens: Optional[int] = 2048, history: Optional[list] = None) -> str:
        """
        生成响应

        参数:
            prompt: 用户输入的提示词
            history: 可选的历史对话列表，用于上下文
        """
        # 构造消息列表，默认包含当前用户输入
        messages = history or []
        messages.append({"role": "user", "content": prompt})

        # 请求参数
        payload = {
            "messages": messages
        }

        # 添加 access_token 到 URL
        url = f"{self.chat_url}?access_token={self.access_token}"

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            # 文心一言的响应中，结果通常在 "result" 字段
            return data["result"]
        except Exception as e:
            print(f"文心一言 API 错误: {e}")
            return ""