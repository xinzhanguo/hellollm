FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

LABEL maintainer="kaixin <e@xinzhanguo.cn>"

RUN pip install tokenizers==0.13.3 transformers==4.30.2 && \
    pip install accelerate -U

WORKDIR /work

COPY text text
COPY sanguo.py sanguo.py
COPY hellollm.py hellollm.py

CMD ["python", "hellollm.py"]
