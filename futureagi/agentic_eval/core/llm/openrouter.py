import os

import anthropic
import boto3
from anthropic import AnthropicBedrock
from botocore.exceptions import ClientError
from dsp import LM
from openai import OpenAI

from agentic_eval.core.utils.model_config import LiteLlmProvider
import structlog

logger = structlog.get_logger(__name__)


class OpenRouter(LM):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        self.max_tokens = kwargs.get("max_tokens", 8196)
        self.temperature = kwargs.get("temperature", 0.5)
        self.history = []
        self.provider = kwargs.get("provider", LiteLlmProvider.ANTHROPIC.value)
        self._init_client()

    def _init_client(self):
        if self.provider == "openai":
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "anthropic":
            self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.provider == "openrouter":
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
        elif self.provider == "aws_bedrock_anthropic":
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            aws_region = os.getenv("AWS_REGION", "us-east-1")
            self.client = AnthropicBedrock(
                aws_access_key=aws_access_key,
                aws_secret_key=aws_secret_key,
                aws_region=aws_region,
            )
        elif self.provider == "aws_bedrock":
            aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
            aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
            # aws_region = os.getenv("AWS_REGION", "us-east-1")
            aws_region = "us-west-2"
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=aws_region,
            )

            self.client = session.client("bedrock-runtime")

    def basic_request(self, prompt: str, **kwargs):
        if self.provider in ["openai", "openrouter"]:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            if not getattr(completion, "choices", None):
                raise ValueError("LLM provider returned no choices for basic_request")

            try:
                response = completion.choices[0].message.content
            except Exception as e:
                logger.error("error", e)
                response = ""

            return response
        elif self.provider in ["anthropic", "aws_bedrock_anthropic"]:
            response = self.client.messages.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            try:
                return response.content[0].text
            except Exception as e:
                logger.error("error", e, response.error)
                return ""
        elif self.provider in ["aws_bedrock"]:
            conversation = [
                {
                    "role": "user",
                    "content": [{"text": prompt}],
                }
            ]

            response = self.client.converse(
                modelId=self.model_name,
                messages=conversation,
                inferenceConfig={
                    "maxTokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            try:
                # Extract and print the response text.
                response_text = response["output"]["message"]["content"][0]["text"]
                return response_text
            except (ClientError, Exception) as e:
                logger.error(f"ERROR: Can't invoke . Reason: {e}")

    def __call__(self, prompt, **kwargs):
        return [self.basic_request(prompt, **kwargs)]
