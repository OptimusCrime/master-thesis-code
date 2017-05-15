FROM python:3.5.3
MAINTAINER thomas.gautvedt

# Set some env variables
ENV APP_DIR=/home/rorschach
ENV PYTHONPATH=/home/rorschach:$PYTHONPATH

# Change working directory
WORKDIR $APP_DIR

# Install the requirements
COPY requirements.txt $APP_DIR/requirements.txt
RUN pip install -r $APP_DIR/requirements.txt

# Change the matplotlib backend to Agg
RUN sed -i "s/backend\s+: tkagg/backend : Agg/g"  /usr/local/lib/python3.5/site-packages/matplotlib/mpl-data/matplotlibrc

# Default run bash
CMD ["bash"]
