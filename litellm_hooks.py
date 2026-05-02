"""LiteLLM pre-call hooks for this project.

cch_stripper removes Claude Code's `x-anthropic-billing-header` text items from
the request before forwarding to vLLM. The header contains a per-request hash
(`cch=<hex>`) that varies on every turn — when present at the very start of the
system content, it defeats vLLM's prefix cache for the entire downstream prompt
(observed: 0.3% hit rate without this hook). Stripping it has no functional
impact since vLLM is not the Anthropic billing system.
"""

import sys

from litellm.integrations.custom_logger import CustomLogger


def _log(msg):
    print(f"[cch_stripper] {msg}", file=sys.stderr, flush=True)


_log("module imported")


class CCHStripper(CustomLogger):
    BILLING_PREFIX = "x-anthropic-billing-header:"

    def __init__(self):
        super().__init__()
        _log("CCHStripper instantiated")

    def _strip_content(self, content):
        """Return (new_content, stripped_count) for either a list of content
        items (Anthropic/OpenAI multimodal) or a plain string."""
        if isinstance(content, list):
            stripped = 0
            new_content = []
            for item in content:
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                    and isinstance(item.get("text"), str)
                    and item["text"].lstrip().startswith(self.BILLING_PREFIX)
                ):
                    stripped += 1
                    continue
                new_content.append(item)
            return new_content, stripped
        if isinstance(content, str) and content.lstrip().startswith(self.BILLING_PREFIX):
            return "", 1
        return content, 0

    async def async_pre_call_hook(self, user_api_key_dict, cache, data, call_type):
        if not isinstance(data, dict):
            return data
        stripped = 0

        # Anthropic format: system prompt is top-level `data["system"]`
        # (string or list of content blocks). This is where Claude Code's
        # billing header lives.
        system = data.get("system")
        if system is not None:
            new_system, n = self._strip_content(system)
            data["system"] = new_system
            stripped += n

        # OpenAI format (after conversion): system prompt is in messages[]
        # with role=='system'. Also scan all messages defensively.
        messages = data.get("messages")
        if isinstance(messages, list):
            for msg in messages:
                if not isinstance(msg, dict):
                    continue
                new_content, n = self._strip_content(msg.get("content"))
                msg["content"] = new_content
                stripped += n

        if stripped:
            _log(f"pre_call: stripped {stripped} billing-header item(s) (call_type={call_type})")
        return data


cch_stripper = CCHStripper()
_log("cch_stripper instance created")
