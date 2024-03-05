FROM python:3.9

RUN git clone --depth 1 https://github.com/zhengr/SmartSearch.git
RUN wget https://github.com/zhengr/SmartSearch/releases/download/0.1.0/ui.zip && unzip ui.zip

WORKDIR /SmartSearch

COPY ./entrypoint.sh /SmartSearch/entrypoint.sh

COPY ./requirements.txt /SmartSearch/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /SmartSearch/requirements.txt

COPY . .

RUN chmod +x /SmartSearch/entrypoint.sh
ENTRYPOINT ["sh", "/SmartSearch/entrypoint.sh"]

EXPOSE 8081
