# Gan Handwritten Letters Generator
This is a simple project to demonstrate how to apply Mlflow to a Deep learning project, using PyTorch.

It tracks the information of the training process also help simplify the deployment process and also help monitoring that process.
You can download the runs file and mlflow to open the mlflow ui on your computer to see how the results go.

This project is the one use for final of the subject Machine Learning and Artificial Intelligience at Ho Chi Minh City University of Technology and Education. Got 9.5/10 marks (A+).

Some main scripts you need to know is the scripts that are not put in a folder.
1. deploy.py
   This is called batch inference. Which mean you gotta download the model first then use it like a normal model you create using you physical machine.
3. early_best_weights.py
   This is also for training the model, but this more advance, with apply Early Stopping and Learning Rate Scheduler.
   This included system metrics.
4. online_inference.py
   This show how you can deploy the model using the server, you can send a payload to the Mlflow server and get the results back, no need to download the model.
5. runner.py
   This is just train the model normally, track it with Mlflow, but no system metrics.
