---
paths:
  - "**/api*"
  - "**/server*"
  - "**/webhook*"
  - "**/auth*"
  - "**/routes*"
  - "**/middleware*"
---
# Security Rules

- Validate all inputs at the API boundary: max length, max size, character set constraints
- Prompt injection is a real attack surface — system prompt content must never be controllable by request payload
- Never log raw user input, API keys, or tokens at INFO level or above — log a hash or truncated prefix
- Rate-limit all API endpoints by default
- Set and enforce per-request timeouts on every operation
- Verify all webhook signatures (HMAC-SHA256 + constant-time comparison)
- Never store API keys or tokens in the codebase — use environment variables
- Validate all JWT claims and enforce tenant isolation in multi-tenant systems
