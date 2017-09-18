FROM crestedibis/python3-opencv3-docker

RUN pip install \
        pandas \
        tqdm \
        pyyaml \
        sklearn \
        scipy

COPY . /usr/workspace
WORKDIR /usr/workspace

CMD ["bash"]
