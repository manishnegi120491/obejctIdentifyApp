#!/bin/bash
echo "Checking for Docker files..."
find . -name "Dockerfile*" -type f
find . -name "docker-compose*" -type f
find . -name ".dockerignore" -type f
echo "Docker file check complete."
