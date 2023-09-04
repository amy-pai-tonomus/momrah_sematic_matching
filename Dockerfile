##################
FROM openjdk:8

RUN apt-get update
RUN apt-get install -y wget
RUN apt-get install -y git 
RUN apt-get install -y curl
RUN apt-get install -y vim
RUN apt-get install -y tar
RUN apt-get install -y bzip2

RUN apt-get update
RUN apt-get install -y python3-dev
RUN apt-get install -y python3-pip

RUN pip3 install Flask==2.2.2
RUN pip3 install flasgger==0.9.5
RUN pip3 install Werkzeug==2.2.2
RUN pip3 install flask-restx==1.0.3
RUN pip3 install pandas==2.0.3
RUN pip3 install torch==2.0.1
RUN pip3 install transformers==4.31.0
RUN pip3 install sentence_transformers==2.2.2

WORKDIR /root

RUN python3 -c "from sentence_transformers import SentenceTransformer;model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

RUN echo "1dg1s5g1s5g"
COPY *.py  /root/
COPY *.json  /root/
COPY *.conf  /root/

CMD python3 app_path.py
##################
