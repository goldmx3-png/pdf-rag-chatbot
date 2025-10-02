#!/bin/bash

# PDF RAG Chatbot - Docker Development Scripts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker and Docker Compose are installed
check_dependencies() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
}

# Create .env file from .env.example if it doesn't exist
setup_env() {
    if [ ! -f .env ]; then
        if [ -f .env.example ]; then
            cp .env.example .env
            print_success "Created .env file from .env.example"
            print_warning "Please update .env file with your configuration before starting services"
        else
            print_error ".env.example file not found"
            exit 1
        fi
    fi
}

# Start all services
start_services() {
    print_status "Starting PDF RAG Chatbot services..."
    check_dependencies
    setup_env
    
    # Use docker compose or docker-compose based on availability
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD up -d
    
    if [ $? -eq 0 ]; then
        print_success "All services started successfully!"
        echo ""
        print_status "Services running at:"
        echo "  • Frontend: http://localhost:3000"
        echo "  • Backend API: http://localhost:8080"
        echo "  • ChromaDB: http://localhost:8000"
        echo "  • MongoDB: localhost:27017"
        echo ""
        print_status "To check service status: ./dev-scripts.sh status"
        print_status "To view logs: ./dev-scripts.sh logs"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Stop all services
stop_services() {
    print_status "Stopping PDF RAG Chatbot services..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD down
    print_success "All services stopped"
}

# Restart all services
restart_services() {
    print_status "Restarting PDF RAG Chatbot services..."
    stop_services
    start_services
}

# Show service status
show_status() {
    print_status "Service Status:"
    docker ps --filter "name=pdf-rag" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
}

# Show logs
show_logs() {
    if [ -z "$2" ]; then
        print_status "Showing logs for all services..."
        if docker compose version &> /dev/null; then
            docker compose logs -f
        else
            docker-compose logs -f
        fi
    else
        print_status "Showing logs for $2..."
        if docker compose version &> /dev/null; then
            docker compose logs -f "$2"
        else
            docker-compose logs -f "$2"
        fi
    fi
}

# Rebuild services
rebuild_services() {
    print_status "Rebuilding PDF RAG Chatbot services..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD down
    $COMPOSE_CMD build --no-cache
    $COMPOSE_CMD up -d
    
    print_success "Services rebuilt and restarted"
}

# Clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    
    if docker compose version &> /dev/null; then
        COMPOSE_CMD="docker compose"
    else
        COMPOSE_CMD="docker-compose"
    fi
    
    $COMPOSE_CMD down -v --remove-orphans
    docker system prune -f
    
    print_success "Cleanup completed"
}

# Show help
show_help() {
    echo "PDF RAG Chatbot - Docker Development Scripts"
    echo ""
    echo "Usage: ./dev-scripts.sh [command]"
    echo ""
    echo "Commands:"
    echo "  start     Start all services"
    echo "  stop      Stop all services"
    echo "  restart   Restart all services"
    echo "  status    Show service status"
    echo "  logs      Show logs for all services"
    echo "  logs [service]  Show logs for specific service"
    echo "  rebuild   Rebuild and restart all services"
    echo "  cleanup   Stop services and clean up Docker resources"
    echo "  help      Show this help message"
    echo ""
    echo "Available services: mongodb, chromadb, backend, frontend"
}

# Main script logic
case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs "$@"
        ;;
    rebuild)
        rebuild_services
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac