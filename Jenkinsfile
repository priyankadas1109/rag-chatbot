pipeline {
  agent any

  environment {
    AWS_REGION    = 'us-east-1'
    IMAGE_NAME    = 'pdf-bot'
    IMAGE_TAG     = 'latest'
    // ECR_REPO uses the AWS_ACCOUNT_ID defined securely below
  }

  stages {
    stage('Build Docker Image') {
      steps {
        script {
          dockerImage = docker.build("${IMAGE_NAME}:${IMAGE_TAG}", "-f pdf_bot.Dockerfile .")
        }
      }
    }

    stage('Push to ECR') {
      steps {
        withCredentials([
          string(credentialsId: 'aws-account-id', variable: 'AWS_ACCOUNT_ID'),
          [$class: 'AmazonWebServicesCredentialsBinding', credentialsId: 'aws-creds']
        ]) {
          script {
            def ecrRepo = "${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
            sh """
              aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ecrRepo
              docker tag ${IMAGE_NAME}:${IMAGE_TAG} $ecrRepo/${IMAGE_NAME}:${IMAGE_TAG}
              docker push $ecrRepo/${IMAGE_NAME}:${IMAGE_TAG}
            """
          }
        }
      }
    }

    stage('Post-Deploy Actions') {
      steps {
        echo '✅ Docker image pushed to ECR successfully.'
        sh 'docker image prune -f'
      }
    }
  }
}
