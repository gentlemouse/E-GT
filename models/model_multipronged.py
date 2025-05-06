from models.base_model import BaseModel
from openai import OpenAI
import os
import requests
import json
from typing import Optional, Dict, List, Any


# 1. OpenAIModel
class OpenAIModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "gpt-3.5-turbo"):
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_response(self, prompt: str, temperature: float = 0.3, reset_history: bool = False) -> str:
        if reset_history:
            self.history = self.history[:1]  # 保留系统消息
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                temperature=temperature
            )
            result = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            if "429" in str(e):
                print("OpenAI API Error: Quota exceeded. Check your account balance or upgrade your plan.")
            elif "requests.exceptions.SSLError" in str(e):
                print("OpenAI API Error: SSL certificate verification failed. Check your network proxy settings.")
            else:
                print(f"OpenAI API Error: {e}")
            return ""

# 2. DeepSeekModel
class DeepSeekModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "DeepSeek-R1"):
        super().__init__(api_key, model_name)
        self.base_url = "https://infer-modelarts-cn-southwest-2.modelarts-infer.com/v1/infers/c3cfa9e2-40c9-485f-a747-caae405296ef/v1/chat/completions"
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_response(self, prompt: str, system_prompt: str = "", max_tokens: int = 1024,
                         temperature: float = 0.3, stream: bool = False, reset_history: bool = False, **kwargs) -> str:
        if system_prompt:
            self.history[0]["content"] = system_prompt
        if reset_history:
            self.history = [self.history[0]]  # 保留系统消息
        self.history.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": self.history,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        try:
            response = requests.post(self.base_url, headers=headers, data=json.dumps(payload), verify=False, timeout=30)
            if response.status_code != 200:
                raise Exception(f"API Error [{response.status_code}]: {response.text}")
            response_data = response.json()
            if 'choices' in response_data and len(response_data['choices']) > 0:
                result = response_data['choices'][0]['message']['content']
                self.history.append({"role": "assistant", "content": result})
                return result
            return ""
        except requests.exceptions.RequestException as e:
            print(f"DeepSeek API Request Failed: {str(e)}")
            return ""
        except json.JSONDecodeError:
            print(f"Invalid API Response: {response.text}")
            return ""
        except Exception as e:
            print(f"DeepSeek API Error: {str(e)}")
            return ""

# 3. GeminiModel
class GeminiModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-pro"):
        super().__init__(api_key, model_name)
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        self.history = [{"role": "system", "parts": [{"text": "You are a helpful assistant."}]}]

    def generate_response(self, prompt: str, temperature: Optional[float] = 0.3,
                         max_output_tokens: Optional[int] = 1024, safety_settings: Optional[Dict] = None,
                         reset_history: bool = False, **kwargs) -> str:
        if reset_history:
            self.history = [self.history[0]]  # 保留系统消息
        user_message = {"role": "user", "parts": [{"text": prompt}]}
        self.history.append(user_message)
        headers = {"Content-Type": "application/json"}
        payload = {"contents": self.history, "generationConfig": {}}
        if temperature is not None:
            payload["generationConfig"]["temperature"] = temperature
        if max_output_tokens is not None:
            payload["generationConfig"]["maxOutputTokens"] = max_output_tokens
        if safety_settings:
            payload["safetySettings"] = safety_settings
        if kwargs:
            payload.update(kwargs)
        try:
            response = requests.post(url=self.base_url, params={"key": self.api_key}, headers=headers, json=payload, timeout=30)
            if response.status_code != 200:
                error_msg = response.json().get("error", {}).get("message", "Unknown error")
                raise Exception(f"API Error [{response.status_code}]: {error_msg}")
            response_data = response.json()
            if "candidates" in response_data and len(response_data["candidates"]) > 0:
                assistant_response = response_data["candidates"][0]["content"]["parts"][0]["text"]
                assistant_message = {"role": "assistant", "parts": [{"text": assistant_response}]}
                self.history.append(assistant_message)
                return assistant_response
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

# 4. KimiModel (未修改，已支持多轮对话)
class KimiModel(BaseModel):
    def __init__(self, api_key: str = None, model_name: str = "moonshot-v1-8k"):
        api_key = api_key or os.environ.get("MOONSHOT_API_KEY")
        if not api_key:
            raise ValueError("API Key not provided. Set MOONSHOT_API_KEY or pass api_key.")
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://api.moonshot.cn/v1")
        self.history = [{"role": "system", "content": "你好"}]

    def generate_response(self, prompt: str, temperature: float = 0.3, reset_history: bool = False) -> str:
        if reset_history:
            self.history = self.history[:1]
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                temperature=temperature
            )
            result = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            print(f"Kimi API Error: {e}")
            return ""

    def get_history(self) -> List[Dict[str, str]]:
        return self.history

    def clear_history(self):
        self.history = self.history[:1]

# 5. ZhipuModel
class ZhipuModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "glm-4v-flash"):
        super().__init__(api_key, model_name)
        self.base_url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_response(self, prompt: str, temperature: float = 0.3, max_tokens: int = 1024, reset_history: bool = False) -> str:
        if reset_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model_name,
            "messages": self.history,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            if "choices" not in data or not data["choices"]:
                raise ValueError("API response missing 'choices' field")
            result = data["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": result})
            return result
        except requests.exceptions.RequestException as e:
            print(f"Zhipu API HTTP Error: {e}")
            return ""
        except (KeyError, ValueError) as e:
            print(f"Zhipu API Response Parsing Error: {e}")
            return ""
        except Exception as e:
            print(f"Zhipu API Unexpected Error: {e}")
            return ""

# 6. DoubaoModel
class DoubaoModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "doubao-1-5-lite-32k-250115"):
        super().__init__(api_key, model_name)
        self.base_url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_response(self, prompt: str, temperature: Optional[float] = 0.3, max_tokens: Optional[int] = 1024,
                         reset_history: bool = False) -> str:
        if reset_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {"model": self.model_name, "messages": self.history}
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            result = response.json()["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            print(f"Doubao API Error: {e}")
            return ""

# 7. XunfeiModel
class XunfeiModel(BaseModel):
    def __init__(self, api_key: str, model_name: str = "lite"):
        super().__init__(api_key, model_name)
        self.base_url = "https://spark-api-open.xf-yun.com/v1/chat/completions"
        self.history = [{"role": "system", "content": "你是知识渊博的助理"}]

    def generate_response(self, prompt: str, temperature: Optional[float] = 0.3, max_tokens: Optional[int] = 1024,
                         stream: bool = False, reset_history: bool = False, **kwargs) -> str:
        if reset_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model_name,
            "messages": self.history,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            result = data["choices"][0]["message"]["content"]
            self.history.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            print(f"Xunfei API Error: {str(e)}")
            return ""

# 8. HunyuanModel
class HunyuanModel(BaseModel):
    def __init__(self, api_key: str = None, model_name: str = "hunyuan-lite"):
        api_key = api_key or os.environ.get("HUNYUAN_KEY")
        if not api_key:
            raise ValueError("API Key not provided. Set HUNYUAN_API_KEY or pass api_key.")
        super().__init__(api_key, model_name)
        self.client = OpenAI(api_key=api_key, base_url="https://api.hunyuan.cloud.tencent.com/v1")
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

    def generate_response(self, prompt: str, enable_enhancement: bool = True, temperature: Optional[float] = 0.3,
                         max_tokens: Optional[int] = 1024, reset_history: bool = False, **kwargs) -> str:
        if reset_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=self.history,
                extra_body={"enable_enhancement": enable_enhancement},
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            result = response.choices[0].message.content
            self.history.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            print(f"Hunyuan API Error: {e}")
            return ""

# 9. WenxinModel
class WenxinModel(BaseModel):
    def __init__(self, api_key: str, secret_key: str, model_name: str = "ernie-4.0-turbo-8k"):
        super().__init__(api_key, model_name)
        self.secret_key = secret_key
        self.token_url = "https://aip.baidubce.com/oauth/2.0/token"
        self.chat_url = f"https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/{model_name}"
        self.access_token = self._get_access_token()
        self.history = [{"role": "system", "content": "You are a helpful assistant."}]

    def _get_access_token(self) -> str:
        params = {"grant_type": "client_credentials", "client_id": self.api_key, "client_secret": self.secret_key}
        try:
            response = requests.get(self.token_url, params=params)
            response.raise_for_status()
            data = response.json()
            return data["access_token"]
        except Exception as e:
            print(f"Wenxin access_token fetch failed: {e}")
            raise

    def generate_response(self, prompt: str, temperature: Optional[float] = 0.3,
                         max_tokens: Optional[int] = 1024, reset_history: bool = False) -> str:
        if reset_history:
            self.history = [self.history[0]]
        self.history.append({"role": "user", "content": prompt})
        url = f"{self.chat_url}?access_token={self.access_token}"
        payload = {"messages": self.history}
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            data = response.json()
            result = data["result"]
            self.history.append({"role": "assistant", "content": result})
            return result
        except Exception as e:
            print(f"Wenxin API Error: {e}")
            return ""