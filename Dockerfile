FROM python:3.11-slim

WORKDIR /opt/app
COPY . .

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter && \
    apt-get install build-essential -yq && \
    apt-get install -yq gcc g++ 


RUN pip install --no-cache-dir -r /opt/app/requirements.txt

EXPOSE 7860
CMD bash -c "python /opt/app/app.py"
