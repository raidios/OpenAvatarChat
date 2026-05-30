from dataclasses import dataclass
import re
from typing import Literal, Optional


from engine_utils.media_utils import ImageUtils


@dataclass
class HistoryMessage:
    role: Optional[Literal['avatar', 'human']] = None
    content: str = ''
    timestamp: Optional[str] = None


name_dict = {
    "avatar": "assistant",
    "human": "user"
}


# 仅用于历史回写：把已知的情绪动作标签 [happy]/[shy]/[apologize]/[scared] 等剥掉，
# 避免污染 LLM 的上下文。保留未知方括号片段（路名/缩写）原样。
_ACTION_TAG_HISTORY_RE = re.compile(
    r"\[\s*(?:happy|shy|apologize|scared)\s*\]",
    flags=re.IGNORECASE,
)


def strip_action_tags(text: str) -> str:
    if not text:
        return text
    return _ACTION_TAG_HISTORY_RE.sub("", text)


def filter_text(text):
    pattern = r"[^a-zA-Z0-9\u4e00-\u9fff,.\~!?，。！？ ]"  # 匹配不在范围内的字符
    filtered_text = re.sub(pattern, "", text)
    return filtered_text


class ChatHistory:
    def __init__(self, history_length):
        self.max_history_length = history_length
        self.message_history = []

    def add_message(self, message: HistoryMessage):
        history = self.message_history
        history.append(message)
        # thread safe
        while len(history) >= self.max_history_length:
            history.pop(0)

    def generate_next_messages(self, chat_text, images):
        def history_to_message(history: HistoryMessage):
            return {
                "role": name_dict[history.role],
                "content": filter_text(strip_action_tags(history.content)),
            }
        history = self.message_history
        messages = list(map(history_to_message, history))
        if images and len(images) > 0:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": filter_text(chat_text),
                    },
                ] + (list(map(lambda x: {"type": "image_url", "image_url": {"url": ImageUtils.format_image(x)}}, images)))
            })
        else: 
            messages.append({
                "role": "user",
                "content": filter_text(chat_text),
            })
        return messages        
    

  