AI Based Image Retrieval

Instructions for deployment on GPU machine or server:
- Clone the repository
- Create conda environment from environment.yml file
- Go to --> deployment_latest --> run app.py and the file will run on the local server


## Deployment Instructions
Github
- Repository Name: augier-prediction-model
- Repository URL: https://github.com/mkhb654/augier-prediction-model.git 
- Clone URL: git clone: https://github.com/mkhb654/augier-prediction-model.git 
## Local Machine Setup
- Git clone: https://github.com/mkhb654/augier-prediction-model.git 
- cd augier-prediction-model
- Create conda environment from the .yml file in the folder: 
```
conda env create -f environment.yml 
```
Confirm if the environment with the name ‘pytorch’ is created: 
```
conda env list
```
If the environment is present then activate the environment: conda activate pytorch
* cd deployment_latest
* Run command: python app.py 
The application will run on the local server and will now respond to postman requests. 
## Server Machine Setup
Login to the server using the .pem file. Place the .pem file in a folder and open the terminal in that folder. Run the following commands one by one: 
chmod 400 Image_retrieval.pem
```
ssh -i “Image_retrieval.pem” ubuntu@ec2-35-175-222-98.compute-1.amazonawas.com 
```
You will be logged into the ec2 instance. Now you have to git clone the repository: https://github.com/mkhb654/augier-prediction-model.git 
```
cd augier-prediction-model
```
Create conda environment from the .yml file in the folder: 
```
conda env create -f environment.yml 
```
Confirm if the environment with the name ‘pytorch’ is created: 
```
conda env list
```
If the environment is present then activate the environment: conda activate pytorch
```
cd deployment_latest
Run command: python app.py
``` 
The application will run on the server and will now respond to postman requests. 
##  Code files
```
The app.py (→  aurgier-prediction-model → deployment_latest → app.py ) file is the api file. It can opened in a text editor like VS code on a local machine and can be opened on the server machine using the command: nano app.py
``` 







