from openai import OpenAI
import os

# Configuration from provided file
LOCAL_API_BASE = ""
API_KEY = ""
MODEL_NAME = ""

class LLM:
    def __init__(self, model_name=MODEL_NAME, api_key=API_KEY, api_base=LOCAL_API_BASE):
        self.model_name = model_name
        self.api_key = api_key
        self.api_base = api_base
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    def generate_response(self, user_prompt, max_tokens=8192, stop=["null"], temperature=0.7, return_usage=False):
        # Qwen3 specific instruction
        if "qwen3" in self.model_name.lower():
            user_prompt += '/think'
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": user_prompt,
                    }
                ],
                stream=False,
                max_tokens=max_tokens,
                stop=stop,
                temperature=temperature,
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if return_usage and response.usage:
                    usage = f"Input: {response.usage.prompt_tokens}, Output: {response.usage.completion_tokens}"
                    return content, usage
                return content, None
            else:
                return "Error: No response content", None
        except Exception as e:
            return f"Error: {str(e)}", None

# Initialize default instance
llm_client = LLM()

