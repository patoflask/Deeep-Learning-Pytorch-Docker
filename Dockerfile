FROM python:3
RUN adduser ptok

WORKDIR /home/ptok

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt
RUN pip3 install gunicorn

COPY app app
COPY utils utils
COPY modeldir modeldir
COPY entrypoint.sh entrypoint.sh

RUN chmod +x ./entrypoint.sh
RUN chown -R ptok:ptok ./

USER ptok

EXPOSE 5001

ENTRYPOINT ["sh", "entrypoint.sh"]

