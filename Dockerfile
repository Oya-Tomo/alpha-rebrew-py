FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN apt update

RUN apt install vim \
        curl \
        python3 \
        python3-pip \
        -y

WORKDIR /workspace

CMD [ "python3", "src/train.py" ]