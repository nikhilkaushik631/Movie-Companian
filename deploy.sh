#!/bin/bash

# Movie Companion Deployment Script
# This script builds and deploys the application using Docker Compose

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="movie-companion"
ENVIRONMENT=${1:-"development"}
COMPOSE_FILE="docker-compose.yml"

if [ "$ENVIRONMENT" = "production" ]; then
    COMPOSE_FILE="docker-compose.production.yml"
fi

# Functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    success "Prerequisites check passed"
}

# Environment setup
setup_environment() {
    log "Setting up environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        warning ".env file not found. Creating from template..."
        cp .env.template .env
        warning "Please update the .env file with your API keys before proceeding."
        read -p "Press Enter to continue after updating .env file..."
    fi
    
    # Create necessary directories
    mkdir -p logs ssl-certs
    
    success "Environment setup completed"
}

# Build and deploy
deploy() {
    log "Starting deployment for $ENVIRONMENT environment..."
    
    # Stop existing containers
    log "Stopping existing containers..."
    docker-compose -f $COMPOSE_FILE down 2>/dev/null || true
    
    # Build and start services
    log "Building and starting services..."
    docker-compose -f $COMPOSE_FILE up --build -d
    
    # Wait for services to be ready
    log "Waiting for services to start..."
    sleep 30
    
    # Health checks
    log "Performing health checks..."
    
    # Check backend health
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        success "Backend is healthy"
    else
        error "Backend health check failed"
        docker-compose -f $COMPOSE_FILE logs backend
        exit 1
    fi
    
    # Check frontend health  
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        success "Frontend is healthy"
    else
        warning "Frontend might still be starting up..."
    fi
    
    success "Deployment completed successfully!"
}

# Show logs
show_logs() {
    log "Showing service logs..."
    docker-compose -f $COMPOSE_FILE logs -f
}

# Cleanup
cleanup() {
    log "Cleaning up..."
    docker-compose -f $COMPOSE_FILE down
    docker system prune -f
    success "Cleanup completed"
}

# Main execution
case "${2:-deploy}" in
    "deploy")
        check_prerequisites
        setup_environment
        deploy
        ;;
    "logs")
        show_logs
        ;;
    "cleanup")
        cleanup
        ;;
    "restart")
        log "Restarting services..."
        docker-compose -f $COMPOSE_FILE restart
        success "Services restarted"
        ;;
    *)
        echo "Usage: $0 [development|production] [deploy|logs|cleanup|restart]"
        echo ""
        echo "Commands:"
        echo "  deploy   - Build and deploy the application (default)"
        echo "  logs     - Show service logs"
        echo "  cleanup  - Stop services and clean up"
        echo "  restart  - Restart all services"
        echo ""
        echo "Examples:"
        echo "  $0                           # Deploy in development mode"
        echo "  $0 production deploy         # Deploy in production mode"
        echo "  $0 production logs           # Show production logs"
        echo "  $0 development cleanup       # Clean up development environment"
        exit 1
        ;;
esac

if [ "${2:-deploy}" = "deploy" ]; then
    echo ""
    log "🎉 Movie Companion is now running!"
    echo ""
    echo "  Frontend: http://localhost:3000"
    echo "  Backend:  http://localhost:8000"
    echo "  API Docs: http://localhost:8000/docs"
    echo ""
    log "To view logs: $0 $ENVIRONMENT logs"
    log "To cleanup:   $0 $ENVIRONMENT cleanup"
fi