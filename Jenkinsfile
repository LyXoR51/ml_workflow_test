pipeline {
    agent any

    parameters {
        string(name: 'FILENAME', defaultValue: '', description: 'Nom du fichier CSV à télécharger')
    }

    environment {
        AWS_ACCESS_KEY_ID = credentials('AWS_ACCESS_KEY_ID')       // Jenkins credentials ID
        AWS_SECRET_ACCESS_KEY = credentials('AWS_SECRET_ACCESS_KEY')
        AWS_DEFAULT_REGION = 'eu-west-3'                           // adapte ta région
        S3_BUCKET = credentials('S3_BUCKET')
        S3_FOLDER = credentials('S3_FOLDER')
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the repository
                git branch: 'main', url: 'https://github.com/LyXoR51/ml_workflow_test.git'
            }
        }

        stage('Download CSV from S3') {
            steps {
                script {
                    if (!params.FILENAME) {
                        error("Le paramètre FILENAME est obligatoire.")
                    }

                    withEnv(["FILENAME=${params.FILENAME}"]) {
                        sh '''
                            python3 -c "
import os
import boto3
import botocore

aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
region_name = os.environ['AWS_DEFAULT_REGION']
bucket = os.environ['S3_BUCKET']
folder = os.environ['S3_FOLDER']
filename = os.environ['FILENAME']

s3 = boto3.client('s3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region_name)

key = folder + filename

try:
    s3.download_file(bucket, key, filename)
    print(f'Fichier téléchargé : {filename}')
except botocore.exceptions.ClientError as e:
    print(f'Erreur lors du téléchargement : {e}')
    exit(1)
"
                        '''
                    }
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image using the Dockerfile
                    sh 'docker build -t ml-pipeline-image .'
                }
            }
        }

        stage('Run Tests Inside Docker Container') {
            steps {
                withCredentials([
                    string(credentialsId: 'MLFLOW_TRACKING_URI', variable: 'MLFLOW_TRACKING_URI'),
                    string(credentialsId: 'AWS_ACCESS_KEY_ID', variable: 'AWS_ACCESS_KEY_ID'),
                    string(credentialsId: 'AWS_SECRET_ACCESS_KEY', variable: 'AWS_SECRET_ACCESS_KEY'),
                    string(credentialsId: 'BACKEND_STORE_URI', variable: 'BACKEND_STORE_URI'),
                    string(credentialsId: 'ARTIFACT_ROOT', variable: 'ARTIFACT_ROOT')
                ]) {
                    script {
                        // Write environment variables to a temporary file
                        writeFile file: 'env.list', text: '''
MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
BACKEND_STORE_URI=$BACKEND_STORE_URI
ARTIFACT_ROOT=$ARTIFACT_ROOT
'''
                    }

                    // Run a temporary Docker container and pass env variables securely via --env-file
                    sh '''
docker run --rm --env-file env.list \
ml-pipeline-image \
bash -c "pytest --maxfail=1 --disable-warnings"
'''
                }
            }
        }
    }

    post {
        always {
            // Clean up workspace and remove dangling Docker images
            sh 'docker system prune -f'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for errors.'
        }
    }
}