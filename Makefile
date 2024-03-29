.PHONY: all bash prepare build clean down logs restart start status stop

SERVER_SERVICE_NAME = rorschach

all: build start prepare run

bash:
	@docker-compose run --rm $(SERVER_SERVICE_NAME) bash

prepare:
	@docker-compose run --rm $(SERVER_SERVICE_NAME) bash docker/install.sh
	@docker-compose run --rm $(SERVER_SERVICE_NAME) cp docker/config.yaml config/config.yaml

run:
	@docker-compose run --rm $(SERVER_SERVICE_NAME) python rorschach/main.py

build:
	@docker-compose build

clean: stop
	@docker-compose rm --force

down:
	@docker-compose down

logs:
	@docker-compose logs -f

restart: stop start

start:
	@docker-compose up -d

status:
	@docker-compose ps

stop:
	@docker-compose stop
