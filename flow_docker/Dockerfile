FROM nvidia/cudagl:11.3.0-devel-ubuntu20.04
SHELL ["/bin/bash", "-c"]
# Set the timezone info because otherwise tzinfo blocks install 
# flow and ignores the non-interactive frontend command 🤬🤬🤬
RUN ln -snf /usr/share/zoneinfo/America/New_York /etc/localtime && echo "/usr/share/zoneinfo/America/New_York" > /etc/timezone

# Core system packages
RUN apt-get update --fix-missing
RUN apt install -y software-properties-common wget curl gpg gcc git make

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

RUN apt install -y apt-utils

ENV TORCH_CUDA_ARCH_LIST="Ampere;Turing;Pascal"
ENV FORCE_CUDA="1"
RUN conda update -y conda
RUN conda install numpy python=3.10 pytorch==1.12.1 torchvision torchaudio cudatoolkit=11.3 -c pytorch -y
RUN pip install open3d==0.16

RUN conda install termcolor h5py gymnasium

RUN pip install opencv-python

RUN git clone https://github.com/facebookresearch/vip.git /vip
WORKDIR /vip
RUN git checkout 676d24ce08e76475595c9a0167d1fdfdd1da5624
RUN pip install -e .

RUN git clone https://github.com/edwhu/Gymnasium-Robotics.git /project
WORKDIR /project
RUN git checkout 5387ae12782c47581fccc07158b4b2d86d3aefdd
RUN pip install -e .

RUN apt install -y unzip 
RUN git clone https://github.com/princeton-vl/RAFT.git /RAFT
WORKDIR /RAFT
RUN ./download_models.sh

RUN apt update --fix-missing
RUN apt install -y qt5-default
RUN apt install -y nano

# Add the project to the PYTHONPATH so that we can import modules from it
ENV PYTHONPATH=/project:/:/RAFT/core/:${PYTHONPATH}
# Add the modified bashrc to the container so we get a nice prompt and persistent history
COPY bashrc /root/.bashrc
# Set the working directory to the project directory, where we will mount the repo
WORKDIR /project