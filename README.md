# posture-corrector


deployed on render and here's the website ( not succesful cus opencv doesnt work on deployed websites due to website permission access for camera etc so alternatives can be embedded and modified in place of opencv)
https://posture-corrector-frontend.onrender.com/



if u want to run, open vs code and clone repo then


for backend setup

cd backend

python -m venv venv

venv\Scripts\activate

pip install -r requirements.txt

uvicorn app:app --reload --port 8000



for frontend setup in new terminal

cd frontend

npm install

npm run dev
