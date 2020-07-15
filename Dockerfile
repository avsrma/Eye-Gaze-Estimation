FROM python:3.7
COPY requirements.txt /mpiigaze/
WORKDIR /mpiigaze
# instruction to be run during image build
RUN pip install -r requirements.txt
# Copy all the files from current source directory(from your system) to
# Docker container in /mpiigaze directory 
# COPY pytorch_mpiigaze-master/ ./mpiigaze
RUN git clone https://github.com/kroniidvul/mpiigaze_project.git /mpiigaze/mpiigaze_project

