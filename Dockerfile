FROM python:3.11-slim-bullseye

# create file 
RUN mkdir /app

# make working directory app file
WORKDIR /app

# copy files in working directory in docker image
##COPY requirements.txt main.py  randomForest100Model_new.pkl  xgb_adasyn_model.pkl /app/
##copy all files 
COPY . /app/ 




# Gerekli Python paketlerini yükle
RUN pip install -r requirements.txt

# start app with uvicorn
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]


##Terminal CMD
##Create Image
##docker build -t xgboost_adayn_poweruser_image:v1 .
##Create Container -local
##docker run -d -p 8000:8000 --name xgboost_container xgboost_adayn_poweruser_image:v1 
##Create image at dockerhub 
##docker tag xgboost_adayn_poweruser_image:v1 yaseminbellioglu/xgboost_adayn_poweruser_image:v1
##Push Image from local to dockerhub & Create image at docker hub
##docker push yaseminbellioglu/xgboost_adasyn_poweruser_image:v1


