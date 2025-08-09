#!/bin/bash

# MLOps Pipeline Deployment Script
set -e

echo "🚀 Starting Iris Classification MLOps Pipeline Deployment..."

# Configuration
IMAGE_NAME="iris-mlops-pipeline"
CONTAINER_NAME="iris-api"
PORT="8000"

# Function to print colored output
print_status() {
    echo -e "\n\033[1;34m$1\033[0m"
}

print_success() {
    echo -e "\033[1;32m$1\033[0m"
}

print_error() {
    echo -e "\033[1;31m$1\033[0m"
}

# Stop and remove existing container if it exists
print_status "🛑 Stopping existing container..."
if docker ps -q -f name=$CONTAINER_NAME | grep -q .; then
    docker stop $CONTAINER_NAME
    print_success "Container stopped successfully"
else
    echo "No existing container found"
fi

if docker ps -aq -f name=$CONTAINER_NAME | grep -q .; then
    docker rm $CONTAINER_NAME
    print_success "Container removed successfully"
fi

# Build the Docker image
print_status "🔨 Building Docker image..."
docker build -t $IMAGE_NAME:latest .
print_success "Docker image built successfully"

# Run the container
print_status "🚀 Starting new container..."
docker run -d \
    --name $CONTAINER_NAME \
    -p $PORT:8000 \
    --restart unless-stopped \
    $IMAGE_NAME:latest

print_success "Container started successfully"

# Wait for the service to be ready
print_status "⏳ Waiting for service to be ready..."
sleep 10

# Health check
print_status "🔍 Performing health check..."
max_attempts=30
attempt=1

while [ $attempt -le $max_attempts ]; do
    if curl -f -s http://localhost:$PORT/health > /dev/null; then
        print_success "✅ Health check passed!"
        break
    else
        echo "Attempt $attempt/$max_attempts failed, retrying in 2 seconds..."
        sleep 2
        attempt=$((attempt + 1))
    fi
done

if [ $attempt -gt $max_attempts ]; then
    print_error "❌ Health check failed after $max_attempts attempts"
    echo "Container logs:"
    docker logs $CONTAINER_NAME --tail 20
    exit 1
fi

# Display service information
print_status "📋 Service Information:"
echo "Container Name: $CONTAINER_NAME"
echo "Image: $IMAGE_NAME:latest"
echo "Port: $PORT"
echo "Status: $(docker ps --format 'table {{.Status}}' -f name=$CONTAINER_NAME | tail -n +2)"

print_status "🌐 Access URLs:"
echo "API Base URL: http://localhost:$PORT"
echo "Health Check: http://localhost:$PORT/health"
echo "API Documentation: http://localhost:$PORT/docs"
echo "Metrics: http://localhost:$PORT/metrics"

print_status "🧪 Test the API:"
echo "curl -X POST \"http://localhost:$PORT/predict\" \\"
echo "  -H \"Content-Type: application/json\" \\"
echo "  -d '{"
echo "    \"sepal_length\": 5.1,"
echo "    \"sepal_width\": 3.5,"
echo "    \"petal_length\": 1.4,"
echo "    \"petal_width\": 0.2"
echo "  }'"

print_success "🎉 Deployment completed successfully!"
print_success "Your Iris Classification MLOps API is now running!"
