FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-devel

RUN apt update

RUN apt install vim \
        curl \
        python3 \
        python3-pip \
        -y

RUN pip3 install torch torchvision numpy ray

WORKDIR /workspace

CMD [ "python3", "./src/train_v3.py" ]