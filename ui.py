import streamlit as st
from inference import load_model_and_tokenizer, summarize_text
FT_MODEL_PATH = "./peft-text-summary" 

st.title("🧠 LoRA LLM Summarizer")
st.write(
"OpenAI released a new model that significantly improves reasoning and coding. "
"Analysts expect rapid adoption across industries due to better latency and cost. "
"Early users report smoother integrations and fewer hallucinations."
)
col_in, col_out = st.columns([1, 1])
with col_in:
    sample = (
        "OpenAI released a new model that significantly improves reasoning and coding. "
        "Analysts expect rapid adoption across industries due to better latency and cost. "
        "Early users report smoother integrations and fewer hallucinations."
)
    text = st.text_area("Paste text to summarize", value=sample, height=260)

with col_out:
    style = st.selectbox("Summary style", ["bullet points", "concise paragraph", "executive memo", "study notes"])
    target_len = st.slider("Target length (words)", 30, 400, 120, 10)

load_btn = st.button("Load Model", use_container_width=True)

if load_btn:
    try:
        with st.spinner("Loading model & tokenizer..."):
            ft_model, tokenizer = load_model_and_tokenizer(FT_MODEL_PATH)
            st.success("Model loaded successfully!")
            st.session_state["_lora_loaded_"] = True
    except Exception as e:
        st.session_state["_lora_loaded_"] = False
        st.error(f"Failed to load model: {e}")

run_btn = st.button("Summarize", use_container_width=True)

if run_btn:
    if not st.session_state.get("_lora_loaded_"):
        st.warning("Please load the model first from the sidebar.")
    else:
        try:
            ft_model, tokenizer = load_model_and_tokenizer(FT_MODEL_PATH)
            # chunks = chunk_by_tokens(text, tokenizer, max_input_tokens)
            # if not chunks:
            #     st.warning("No input text provided.")
            # else:
            # st.info(f"Input split into {len(chunks)} chunk(s) for processing.")
            # outputs: List[str] = []
            # for idx, ck in enumerate(chunks, 1):
            #     with st.spinner(f"Summarizing chunk {idx}/{len(chunks)}..."):
            out = summarize_text(
                ft_model,
                tokenizer,
                text,
                )
            # outputs.append(out)
            # final = "\n\n".join(outputs)
            st.subheader("Summary")
            st.write(out)

            # Show a tiny report on tokens
            # in_tok = sum(len(tokenizer(c, add_special_tokens=False).input_ids) for c in chunks)
            # st.caption(f"≈ Input tokens: {in_tok} • Output chars: {len(final)}")
        except Exception as e:
            st.error(f"Summarization failed: {e}")