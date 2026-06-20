"""High-level LLM client that routes requests through provider adapters."""

from .llm_types import LLMRequest, LLMResponse
from .provider import ProviderFactory
from .rate_limiter import RateLimiter

try:
    from config.settings import get_settings
except Exception:
    def get_settings():
        return None


class LLMClient:
    """Simple orchestrator for LLM calls.

    All provider-specific logic is encapsulated behind ProviderFactory so the
    rest of the application never imports provider SDKs directly.
    """

    def __init__(self, provider_name: str | None = None, settings=None, rate_limiter: RateLimiter | None = None):
        self.settings = settings or get_settings()
        self.provider_name = provider_name or (getattr(self.settings, "default_provider", None) if self.settings else None) or "openai"
        self.provider = ProviderFactory.get_provider(self.provider_name, settings=self.settings)
        self.rate_limiter = rate_limiter or RateLimiter(getattr(self.settings, "rate_limit_per_minute", 60) if self.settings else 60)

    def chat(self, prompt: str, model: str | None = None, temperature: float = 0.0, max_tokens: int | None = None) -> LLMResponse:
        request = LLMRequest(model=model or getattr(self.settings, "default_model", ""), prompt=prompt, temperature=temperature, max_tokens=max_tokens)
        if not self.rate_limiter.allow():
            raise RuntimeError("Rate limit exceeded")

        response = self.provider.send(request)

        # Validate the response before returning
        try:
            from security.output_validator import validate_output

            validate_output(response.text)
        except Exception:
            # If validation fails, bubble up a runtime error to make failures explicit
            raise

        return response
