# Use a base image with Java 8 JDK
FROM openjdk:8-jdk-slim

# Set the working directory
WORKDIR ./

COPY . ./miningframework

# Set the entry point command to run the application
ENTRYPOINT ["bash"]
CMD ["cd", "miningframework"]
