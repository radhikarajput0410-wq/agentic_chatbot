#  Agentic Chatbot App

An AI-powered Agentic Chatbot built with LangGraph, LangChain, and Streamlit, featuring an automated CI/CD pipeline using GitHub Actions, Docker, Docker Hub, and AWS EC2.



##  Features

-  Agentic AI workflow using LangGraph
-  Interactive Streamlit interface
-  Web search integration with Tavily
-  Weather information support
-  Dockerized application
-  Automated CI/CD with GitHub Actions
-  Self-hosted deployment on AWS EC2



## Tech Stack

- Python
- Streamlit
- LangChain
- LangGraph
- Docker
- GitHub Actions
- AWS EC2
- Docker Hub



## Project Structure


.
├── app.py
├── requirements.txt
├── Dockerfile
├── .github/
│   └── workflows/
├── src/
└── README.md

 ##  CI/CD Deployment

The deployment pipeline is fully automated using GitHub Actions.

Workflow:

1. Push code to GitHub.
2. GitHub Actions builds the Docker image.
3. Pushes the image to Docker Hub.
4. AWS EC2 (self-hosted runner) pulls the latest image.
5. The application is redeployed automatically.



