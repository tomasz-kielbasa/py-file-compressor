import streamlit as st
from io import StringIO 
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import torch
import numpy as np

import numpyAc

st.set_page_config(layout="wide")
device = 'cuda' if torch.cuda.is_available() else 'cpu'

@st.cache_resource
def load_model():
    return AutoModelForCausalLM.from_pretrained(
        "PY007/TinyLlama-1.1B-python-v0.1",
    ).to(device)

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("PY007/TinyLlama-1.1B-python-v0.1")

model = load_model()
tokenizer = load_tokenizer()

st.title('Python file compressor')
encode_col, decode_col = st.columns(2, gap='medium')

@st.cache_data
def encode(text):
    bar = st.progress(0.0)
    codec = numpyAc.arithmeticCoding()
    tokenized = tokenizer(text, return_tensors='pt').input_ids.to(device)
    output = list()
    past_key_values = None

    # We can't run a single pass over all tokens, because
    # we get inconsistent results then
    length = tokenized.shape[1]
    for i in range(length):
        bar.progress(min(((i + 1) + (i + 1) ** 2 / 1000) / (length + length ** 2 // 1000), 1.0))
        with torch.no_grad():
            output_ = model(
                input_ids=tokenized[:, i:i + 1],
                use_cache=True,
                past_key_values=past_key_values
            )
        past_key_values = output_.past_key_values
        logits = output_.logits[0, -1:, :]
        output.append(logits)
    output = torch.cat(output, dim=0)
    output = F.softmax(output, dim=-1)
    tokenized = torch.cat((tokenized.squeeze()[1:], torch.tensor([2], device=device))) # Add EOS
    tokenized = tokenized.type(torch.int16).cpu().numpy()
    byte_stream, _ = codec.encode(output.cpu(), tokenized)
    return byte_stream

@st.cache_data
def decode(byte_stream):
    # Unfortunately progressbar for decoding isn't possible/is hard
    decodec = numpyAc.arithmeticDeCoding(byte_stream, 32_000)
    input_ids = [1]
    past_key_values = None

    while input_ids[-1] != 2:
        with torch.no_grad():
            output = model(
                input_ids=torch.tensor([input_ids[-1:]], device=device),
                use_cache=True,
                past_key_values=past_key_values
            )
        past_key_values = output.past_key_values
        logits = output.logits[0, -1:, :]
        logits = F.softmax(logits, dim=-1).cpu()
        next_token = decodec.decode(logits)
        input_ids.append(next_token)
    return input_ids

with encode_col:
    st.header('Convert your python file to binary.')
    python_file = st.file_uploader("Upload your python file here. I recommend files up to 50-100 lines, so it doesn't take too long.")
    if python_file is not None:
        stringio = StringIO(python_file.getvalue().decode("utf-8"))
        code = stringio.read()
        bytes_stream = encode(code)
        bin_filename = f'{python_file.name.split(".")[0]}.bin'
        st.download_button('Download binary file', bytes_stream, bin_filename)

with decode_col:
    st.header('Convert your binary file to python')
    binary_file = st.file_uploader('Upload your binary file here')
    if binary_file is not None:
        tokens = decode(binary_file.read())
        decompressed = tokenizer.decode(tokens, skip_special_tokens=True)
        py_filename = f'{binary_file.name.split(".")[0]}.py'
        st.download_button('Download python file', decompressed, py_filename)
        st.code(decompressed)