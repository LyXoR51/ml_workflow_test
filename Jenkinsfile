pipeline {
    agent any

    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the repository
                git branch: 'main', url: 'https://github.com/LyXoR51/ml_workflow_test.git'
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
                        writeFile file: 'env.list', text: """
MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI
AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
BACKEND_STORE_URI=$BACKEND_STORE_URI
ARTIFACT_ROOT=$ARTIFACT_ROOT
"""
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
            sh 'rm -f env.list'
        }
        success {
            echo 'Pipeline completed successfully!'
        }
        failure {
            echo 'Pipeline failed. Check logs for errors.'
        }
    }
}