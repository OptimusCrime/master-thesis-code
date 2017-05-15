FROM python:3.5.3
MAINTAINER thomas.gautvedt

ENV APP_DIR=/home/rorschach
ENV PYTHONPATH=/home/rorschach

WORKDIR $APP_DIR

CMD ["/bin/bash"]
