
Project developers
========================================================
Luke Murtagh , 
Diane Jugul Dalyop
=========================================================
The aim
Instructing a computer on how to drive by recordings in the simulator.

- model is trained from the recordings 
- model learns road structure 
- self-driving car drives on its own and the process is reviewed and performance is documented

=====================================================================
Current Project Structure
=====================================================================
self-driving/
├── data/                 
│   ├── driving_log.csv
├── src/
│   ├── data_preprocess.py
│   ├── model.py
│   ├── train.py
│   └── drive.py  #connection scripts
├── models  # the trained model
├── README.md
└── requirements.txt

1.Recording data done
2.Pre-process complete
3.Exploration : Steering angle stage 1 : noticed a bias toward zero steering 

