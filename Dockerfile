FROM python:3.5.3
MAINTAINER thomas.gautvedt

ENV APP_DIR=/home/rorschach
ENV PYTHONPATH=/home/rorschach:$PYTHONPATH

WORKDIR $APP_DIR

COPY requirements.txt $APP_DIR/requirements.txt
RUN pip install -r $APP_DIR/requirements.txt

CMD ["bash"]
