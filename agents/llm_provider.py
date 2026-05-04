"""
agents/llm_provider.py
Factory that returns a LangChain-compatible LLM for the chosen provider.
Now uses Chat models where possible to avoid task mismatches.
"""
from __future__ import annotations

def get_llm(provider: str, api_key: str = "", model: str = ""):
    """
    Return a LangChain BaseLLM / BaseChatModel for the given provider.
    """
    provider = provider.lower().strip()

    # ------------------------------------------------------------------ #
    # HuggingFace Inference API                                            #
    # ------------------------------------------------------------------ #
    if provider == "huggingface":
        # We use ChatHuggingFace to handle conversational/chat models properly
        from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
        
        repo_id = model or "openai-community/gpt2"
        
        llm = HuggingFaceEndpoint(
            repo_id=repo_id,
            huggingfacehub_api_token=api_key,
            # We don't specify task here to let HF infer, or use text-generation
            # and then let ChatHuggingFace wrap it.
            task="text-generation",
            max_new_tokens=512,
            temperature=0.3,
            do_sample=True,
        )
        
        try:
            return ChatHuggingFace(llm=llm)
        except Exception:
            # Fallback for models without chat templates (like gpt2)
            return llm

    # ------------------------------------------------------------------ #
    # Novita AI (via Hugging Face Router)                                  #
    # ------------------------------------------------------------------ #
    elif provider == "novita":
        from langchain_openai import ChatOpenAI
        m = model or "meta-llama/Llama-3.2-1B-Instruct"
        full_model = m if ":" in m else f"{m}:novita"
        return ChatOpenAI(
            model=full_model,
            api_key=api_key,
            base_url="https://router.huggingface.co/v1",
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # OpenAI                                                               #
    # ------------------------------------------------------------------ #
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "gpt-4o",
            api_key=api_key,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # Anthropic Claude                                                     #
    # ------------------------------------------------------------------ #
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(
            model=model or "claude-3-5-sonnet-20241022",
            api_key=api_key,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # Ollama (local)                                                       #
    # ------------------------------------------------------------------ #
    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(
            model=model or "llama3",
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # xAI Grok                                                             #
    # ------------------------------------------------------------------ #
    elif provider == "grok":
        from langchain_xai import ChatXAI
        return ChatXAI(
            model=model or "grok-beta",
            api_key=api_key,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # Featherless AI (via Hugging Face Router)                             #
    # ------------------------------------------------------------------ #
    elif provider == "featherless":
        from langchain_openai import ChatOpenAI
        # Hugging Face Unified Router
        m = model or "Qwen/Qwen2.5-1.5B-Instruct"
        full_model = m if ":" in m else f"{m}:featherless-ai"
        return ChatOpenAI(
            model=full_model,
            api_key=api_key,
            base_url="https://router.huggingface.co/v1",
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # Google Gemini                                                        #
    # ------------------------------------------------------------------ #
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model or "gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # Groq                                                                 #
    # ------------------------------------------------------------------ #
    elif provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or "llama-3.1-70b-versatile",
            api_key=api_key,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # DeepSeek                                                             #
    # ------------------------------------------------------------------ #
    elif provider == "deepseek":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "deepseek-chat",
            api_key=api_key,
            base_url="https://api.deepseek.com",
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # Perplexity                                                           #
    # ------------------------------------------------------------------ #
    elif provider == "perplexity":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=model or "sonar-pro",
            api_key=api_key,
            base_url="https://api.perplexity.ai",
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # LM Studio (local server)                                             #
    # ------------------------------------------------------------------ #
    elif provider == "lm-studio":
        from langchain_openai import ChatOpenAI
        base_url = api_key if (api_key and api_key.startswith("http")) else "http://localhost:1234/v1"
        return ChatOpenAI(
            model=model or "model-identifier",
            api_key="not-needed",
            base_url=base_url,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # vLLM (local/remote server)                                           #
    # ------------------------------------------------------------------ #
    elif provider == "vllm":
        from langchain_openai import ChatOpenAI
        base_url = api_key if (api_key and api_key.startswith("http")) else "http://localhost:8000/v1"
        return ChatOpenAI(
            model=model or "model-identifier",
            api_key="not-needed",
            base_url=base_url,
            temperature=0.3,
        )

    # ------------------------------------------------------------------ #
    # llama.cpp (local server)                                             #
    # ------------------------------------------------------------------ #
    elif provider == "llama-cpp":
        from langchain_openai import ChatOpenAI
        # llama.cpp server is OpenAI-compatible.
        # We allow the user to pass the base URL in the 'api_key' field,
        # or default to the standard http://localhost:8080/v1
        base_url = api_key if (api_key and api_key.startswith("http")) else "http://localhost:8080/v1"
        return ChatOpenAI(
            model=model or "local-model",
            api_key="not-needed",
            base_url=base_url,
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            "Choose from: huggingface, novita, featherless, openai, anthropic, ollama, grok, google, groq, deepseek, perplexity, lm-studio, vllm, llama-cpp"
        )