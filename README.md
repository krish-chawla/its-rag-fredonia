# CIED AI Assistants Portfolio

A multi-page Streamlit app built during my AI/GenAI research internship at SUNY Fredonia's
Center for Innovation & Economic Development (CIED). It shows four ways small businesses and
campus offices can use generative AI for customer service and operations.

**Live demo:** https://its-rag-fredonia.streamlit.app

| Page | What it does | How it works |
|---|---|---|
| 🌷 Fredonia Flowers (fictional) | Customer-service chatbot: price list, orders, delivery, returns, refunds, cancellations, hours, flower care | RAG: LangChain + OpenAI `gpt-4o-mini` + ChromaDB over `company_docs.txt` |
| 💻 SUNY Fredonia ITS Assistant | Answers IT questions: campus WiFi (eduroam), FREDmedia for game consoles, laptop recommendations by major, the FRED Laptop Program, email and MFA while traveling, FortiClient VPN setup, phishing, and the Acceptable Use Policy | RAG with query rewriting over `its_policy.txt` + `its_help_topics.txt` |
| 🩺 Healthcare Staff Scheduler | Builds a weekly doctor and nurse schedule from max hours, preferred shifts, days off, coverage needs, and plain-English requests ("Nora can't work Friday") | The LLM turns free-text requests into structured rules. A Google OR-Tools CP-SAT solver then builds a schedule that meets every hard rule and maximizes preferences and fairness, or explains why no schedule is possible |
| 🧾 Invoice Tracker | Dashboard plus chat for "Has INV-1004 been paid?", "Who requested refunds?", "What's the late penalty?". Businesses can upload their own CSV | All money math is in plain Python (`invoices.py`). The LLM answers by calling tools that read those computed results, so it never invents numbers |

All chat pages share a human-escalation feature (`escalation.py`). When the bot can't answer, or
the user asks for a person, it shows the right contact info.

## Design choices
- **Use the LLM for language, not for math.** The scheduler and invoice pages use the model only
  to understand requests and phrase answers. Constraint solving and fee calculations are done in
  deterministic code that can be tested.
- **Ground answers in source documents.** RAG prompts tell the model to say "I don't have that
  information" rather than guess a price, policy, or procedure.
- The ITS help topics come from SUNY Fredonia's public ITS pages (October 2026) and should be
  re-checked each semester.

## Run locally
```bash
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
streamlit run app.py
```

Self-tests for the non-AI logic:
```bash
python scheduler.py
python invoices.py
```
