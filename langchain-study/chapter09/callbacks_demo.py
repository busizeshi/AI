import logging
import os
import sys
import time
from pathlib import Path

from openai import OpenAI


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


BASE_URL = "https://codex.ysaikeji.cn/v1"
API_KEY_ENV = "K_CODEX"
MODEL = os.environ.get("K_CODEX_MODEL", "gpt-5.4")
LOG_FILE = Path(__file__).with_name("output.log")


class Callback:
    def on_chain_start(self, prompt: str):
        pass

    def on_new_token(self, token: str):
        pass

    def on_chain_end(self, output: str, elapsed: float):
        pass

    def on_error(self, error: Exception):
        pass


class ProgressCallback(Callback):
    def on_chain_start(self, prompt: str):
        print("[progress] chain started")

    def on_chain_end(self, output: str, elapsed: float):
        print(f"[progress] chain finished in {elapsed:.2f}s")

    def on_error(self, error: Exception):
        print(f"[progress] error: {error}")


class ContentCallback(Callback):
    def on_new_token(self, token: str):
        print(token, end="", flush=True)

    def on_chain_end(self, output: str, elapsed: float):
        print("\n[content] output collected")


class TokenCounterCallback(Callback):
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def on_chain_start(self, prompt: str):
        self.prompt_tokens = estimate_tokens(prompt)

    def on_new_token(self, token: str):
        self.completion_tokens += estimate_tokens(token)

    def on_chain_end(self, output: str, elapsed: float):
        total = self.prompt_tokens + self.completion_tokens
        print(
            f"[tokens] prompt={self.prompt_tokens}, "
            f"completion={self.completion_tokens}, total={total}"
        )


class FileLogCallback(Callback):
    def __init__(self, path: Path):
        logging.basicConfig(
            filename=path,
            level=logging.INFO,
            format="%(asctime)s %(levelname)s %(message)s",
            encoding="utf-8",
        )

    def on_chain_start(self, prompt: str):
        logging.info("chain started, prompt=%s", prompt)

    def on_chain_end(self, output: str, elapsed: float):
        logging.info("chain finished, elapsed=%.2fs, output=%s", elapsed, output)

    def on_error(self, error: Exception):
        logging.exception("chain failed: %s", error)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def get_client() -> OpenAI:
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"请先设置环境变量 {API_KEY_ENV}")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def run_chain(prompt: str, callbacks: list[Callback]) -> str:
    start = time.perf_counter()
    output_parts = []

    try:
        for callback in callbacks:
            callback.on_chain_start(prompt)

        with get_client().responses.stream(
            model=MODEL,
            input=prompt,
        ) as stream:
            for event in stream:
                if getattr(event, "type", "") != "response.output_text.delta":
                    continue
                token = getattr(event, "delta", "")
                if not token:
                    continue
                output_parts.append(token)
                for callback in callbacks:
                    callback.on_new_token(token)

        output = "".join(output_parts)
        elapsed = time.perf_counter() - start
        for callback in callbacks:
            callback.on_chain_end(output, elapsed)
        return output
    except Exception as error:
        for callback in callbacks:
            callback.on_error(error)
        raise


def main():
    prompt = "请用 120 字以内说明 LangChain 回调机制的作用。"
    callbacks = [
        ProgressCallback(),
        ContentCallback(),
        TokenCounterCallback(),
        FileLogCallback(LOG_FILE),
    ]
    run_chain(prompt, callbacks)
    print(f"[log] saved to {LOG_FILE}")


if __name__ == "__main__":
    main()

