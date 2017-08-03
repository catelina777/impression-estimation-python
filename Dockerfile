FROM python:3.6.2
COPY . /usr/workplace
WORKDIR /usr/workplace
RUN pip install -r requirements.txt
