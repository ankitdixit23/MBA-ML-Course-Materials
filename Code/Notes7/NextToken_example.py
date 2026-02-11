import math
import pandas as pd
from openai import OpenAI
import random
import os

# Load your OpenAI API key from an environment variable
# You should set this in your shell before running the script:
# export OPENAI_API_KEY="your_api_key_here"
# Alternatively, you can set is a permanent environment variable associated 
# with a conda environment or virtualenv (called myenv here):
# conda activate myenv
# conda env config vars set OPENAI_API_KEY="your_key_here"
# There are other ways as well; see https://platform.openai.com/docs/api-reference/authentication
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Instructions for the AI model
CONTINUE_INSTRUCTIONS = (
    "You are a text-completion engine. "
    "Your output must be a DIRECT continuation of the given text. "
    "Do not reply to the user, do not explain, do not restart. "
    "Continue with natural spacing/punctuation. "
)

client = OpenAI(api_key=OPENAI_API_KEY)

# Functions to look at top-k logprobs for next token prediction
def _get(obj, name, default=None):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, dict):
        return obj.get(name, default)
    return default

def topk_logprobs_once(prompt, model="gpt-4.1-mini", k=20):
    # Fix one reference run (T=1). We'll apply T ourselves.
    resp = client.responses.create(
        model=model,
        instructions=CONTINUE_INSTRUCTIONS,
        input=prompt,
        temperature=1.0,
        top_logprobs=k,
        max_output_tokens=16,
        include=["message.output_text.logprobs"],
    )
    msg = next(item for item in resp.output if _get(item, "type") == "message")
    out_text = next(c for c in _get(msg, "content") if _get(c, "type") == "output_text")
    lp0 = _get(out_text, "logprobs")[0]
    top = _get(lp0, "top_logprobs")

    rows = []
    for d in top:
        tok = _get(d, "token")
        lp  = _get(d, "logprob")  # log P(token | prompt) at reference run
        rows.append((tok, lp))

    # Also compute how much mass is outside top-k under the reference distribution
    p_top = sum(math.exp(lp) for _, lp in rows)
    p_other = max(0.0, 1.0 - p_top)

    return rows, p_other

def softmax_from_logprobs(rows, T):
    # rows are (token, logprob_ref). Treat logprob_ref as scores up to an additive constant.
    # Temperature scaling: score/T then softmax over the displayed set.
    scores = [(tok, lp / T) for tok, lp in rows]
    m = max(s for _, s in scores)
    exps = [(tok, math.exp(s - m)) for tok, s in scores]
    Z = sum(v for _, v in exps)
    return {tok: v / Z for tok, v in exps}

def top_next_token_table_offline(prompt, temps=(0.1,0.2,0.7,1.0,1.5), k=10, model="gpt-4.1-mini"):
    rows, p_other_ref = topk_logprobs_once(prompt, model=model, k=k)

    # build table
    out = []
    for tok, _ in rows:
        r = {"token": tok}
        out.append(r)

    for T in temps:
        probs = softmax_from_logprobs(rows, T)
        for r in out:
            r[f"P~(T={T})"] = probs.get(r["token"], 0.0)

    df = pd.DataFrame(out).sort_values(f"P~(T={temps[2]})", ascending=False)
    return df, p_other_ref

# Example usage
prompt = "Hydrangeas, Pale blue in the "
df, p_other_ref = top_next_token_table_offline(prompt, k=10)
print("Reference OTHER mass (outside top-k at T=1):", p_other_ref)
print(df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))

# Iterative continuation with top-k sampling
def next_token_distribution_continuation(text, model="gpt-4.1-mini", T=0.7, k=10):
    """
    Top-k next-token probabilities for continuing `text`.
    Probabilities returned are exp(logprob)
    """
    resp = client.responses.create(
        model=model,
        instructions=CONTINUE_INSTRUCTIONS,
        input=text,
        temperature=T,
        max_output_tokens=16,
        top_logprobs=k,
        include=["message.output_text.logprobs"],
    )

    msg = next(item for item in resp.output if _get(item, "type") == "message")
    out_text_item = next(c for c in _get(msg, "content") if _get(c, "type") == "output_text")
    logprobs = _get(out_text_item, "logprobs")

    first = logprobs[0]
    top = _get(first, "top_logprobs")

    rows = []
    for d in top:
        tok = _get(d, "token")
        lp  = _get(d, "logprob")
        rows.append({"token": tok, "logprob": lp, "prob": math.exp(lp)})

    rows.sort(key=lambda r: r["prob"], reverse=True)
    return rows

def iterative_continuation(start_text, steps=8, temperature=0.7, k=10, decode="sample", model="gpt-4.1-mini"):
    """
    decode: "greedy" or "sample"
    """
    context = start_text
    print("\nSTART:", context, "\n")

    for step in range(1, steps + 1):
        dist = next_token_distribution_continuation(context, model=model, T=temperature, k=k)

        # show table; repr() reveals leading spaces and token pieces
        df = pd.DataFrame({
            "token": [repr(r["token"]) for r in dist],
            "prob":  [r["prob"] for r in dist],
        })
        print(f"Step {step}: top next-token probs (continuation)  T={temperature}")
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

        if decode == "greedy":
            chosen = dist[0]["token"]
        else:
            toks = [r["token"] for r in dist]
            weights = [r["prob"] for r in dist]
            chosen = random.choices(toks, weights=weights, k=1)[0]

        print(f"\nChosen token: {repr(chosen)}\n")

        # IMPORTANT: append exactly; do not add your own spaces
        context = context + chosen

        print("Updated context:", context)
        print("-" * 60)

    print("\nFINAL GENERATED TEXT:\n", context)

# Try it
iterative_continuation(
    "Hydrangeas, Pale blue in the ",
    steps=3,
    temperature=0.7,
    k=10,
    decode="greedy",
    model="gpt-4.1-mini"
)

# Let's look at what happens if we use all of OpenAI's hidden behavior
def generate_and_show_nextdist(text, model="gpt-4.1-mini", T=0.7, k=10, gen_tokens=8):
    resp = client.responses.create(
        model=model,
        instructions=CONTINUE_INSTRUCTIONS,
        input=text,
        temperature=T,
        max_output_tokens=max(16, gen_tokens),
        top_logprobs=k,
        include=["message.output_text.logprobs"],
    )

    msg = next(item for item in resp.output if _get(item, "type") == "message")
    out_text = next(c for c in _get(msg, "content") if _get(c, "type") == "output_text")

    # Distribution for the *first* generated token
    lp0 = _get(out_text, "logprobs")[0]
    top = _get(lp0, "top_logprobs")
    rows = [{"token": _get(d, "token"), "prob": math.exp(_get(d, "logprob"))} for d in top]
    rows.sort(key=lambda r: r["prob"], reverse=True)

    # Actual generated text (several tokens)
    generated = _get(out_text, "text")  # this is the generated continuation string

    return rows, generated

def iterative_continuation_block(start_text, steps=5, temperature=0.7, k=10, gen_tokens=8, model="gpt-4.1-mini"):
    context = start_text
    print("\nSTART:", context, "\n")

    for step in range(1, steps + 1):
        rows, gen = generate_and_show_nextdist(
            context, model=model, T=temperature, k=k, gen_tokens=gen_tokens
        )

        df = pd.DataFrame({
            "token": [repr(r["token"]) for r in rows],
            "prob":  [r["prob"] for r in rows],
        })
        print(f"Step {step}: top next-token probs (T={temperature})")
        print(df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

        print("\nModel generated:", repr(gen), "\n")
        context = context + gen
        print("Updated context:", context)
        print("-" * 60)

    print("\nFINAL TEXT:\n", context)

iterative_continuation_block(
    "Hydrangeas, Pale blue in the ",
    steps=2,
    temperature=0.7,
    k=10,
    gen_tokens=12,
    model="gpt-4.1-mini"
)
