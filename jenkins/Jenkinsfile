pipeline {
  agent any

  environment {
    AWS_REGION = 'us-east-1'
    AWS_ACCOUNT_ID = credentials('aws-account-id')   // 🔐 Secure Jenkins Credential
    IMAGE_NAME = 'pdf-bot'
    IMAGE_TAG = 'latest'
    ECR_REPO = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
  }

  stages {
    stage('Build Docker Image') {
      steps {
        script {
          docker.build("${IMAGE_NAME}:${IMAGE_TAG}", "--file pdf_bot.Dockerfile .")
        }
      }
    }

    stage('Push to ECR') {
      steps {
        withCredentials([[$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-creds']]) {
          sh '''
            aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_REPO
            docker tag ${IMAGE_NAME}:${IMAGE_TAG} $ECR_REPO/${IMAGE_NAME}:${IMAGE_TAG}
            docker push $ECR_REPO/${IMAGE_NAME}:${IMAGE_TAG}
          '''
        }
      }
    }

    stage('Post-Deploy Actions') {
      steps {
        echo '✅ Docker image pushed to ECR.'
        sh 'docker image prune -f'
      }
    }
  }
}
