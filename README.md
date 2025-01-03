# Project Visual Question Answering Demo

## Description
A simple web demo for Visual Question Answering on Streamlit. 

## How to use
### Option 1: Using conda environment
1. Create new conda environment and install required dependencies:
```
$ conda create -n <env_name> -y python=3.12
$ conda activate <env_name>
$ pip3 install -r requirements.txt
```
2. Host streamlit app
```
$ streamlit run app.py
```
### Option 2: Using docker
1. Build docker image
```
$ docker build -t <tag_name> -f docker/Dockerfile .
```
2. Run a docker contaier
```
$ docker run -p 8501:8501 -it <tag_name>
```