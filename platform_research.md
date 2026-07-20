# Platform Research Notes
## Copilot Studio vs OpenAI — Customer Service Chatbot Use Case
Date: June 28, 2026

## Copilot Studio
- Requires Microsoft account to access — barrier for non-Microsoft businesses
- True no-code: visual drag-and-drop bot builder, no programming required
- Document grounding via Bing Search and Bing Custom Search
- Company data routes through Microsoft/Bing servers — privacy consideration
- Free trial available, paid tier required for production use
- Best for: businesses already in the Microsoft 365 ecosystem

## OpenAI
- Assistants API deprecated as of August 2026 — replaced by Responses API
- No-code option (GPT Builder in ChatGPT) exists but is limited
- Code-based path (API + LangChain) is more powerful and flexible
- Document grounding via custom RAG pipeline — data stays local
- Pay-as-you-go pricing, very low cost for testing
- Best for: developer-led projects needing full control over pipeline

## Comparison Table
| Feature | Copilot Studio | OpenAI + LangChain |
|---|---|---|
| Coding required | No | Yes |
| Document ingestion | Built-in (Bing) | Custom RAG pipeline |
| Data privacy | Cloud (Microsoft) | Local (ChromaDB) |
| Cost model | Subscription | Pay-as-you-go |
| Deployment | Web embed, Teams | Custom (any platform) |
| Ecosystem fit | Microsoft 365 | Standalone / flexible |
| Production readiness | High (no-code) | High (with dev effort) |

## Recommendation for Internship
For our use case — training a chatbot on company documentation for two test 
companies — the OpenAI + LangChain path offers more control, better data 
privacy, and directly aligns with the RAG pipeline already under development. 
Copilot Studio may be worth testing as a no-code comparison point for one of 
the two companies to evaluate ease-of-use differences.