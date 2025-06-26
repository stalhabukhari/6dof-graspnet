# This Dockerfile sets up a iDSDF environment with PyTorch (GPU).

FROM tensorflow/tensorflow:1.12.0-devel-gpu

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-key del 7fa2af80 && \
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub && \
    apt update && \
    apt install -y \
        wget git net-tools vim curl build-essential x11vnc zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev \
        libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev freeglut3-dev && \
        # following is for opencv-python (uncomment if not needed)
        apt install -y --no-install-recommends libglib2.0-0 libxrender1 libxext6 libsm6 libgl1-mesa-glx && \
        # repo dependencies
        apt install -y cmake libqt4-dev qtdeclarative5-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

# install dependencies
RUN cd / && \
    git clone https://github.com/PySide/BuildScripts && \
    cd BuildScripts && \
    git submodule init && \
    git submodule update && \
    git submodule foreach git checkout master && \
    git submodule foreach git pull && \
    ./dependencies.ubuntu.sh || true && \
    ./build_and_install

RUN cd / && \
    git clone https://github.com/charlesq34/pointnet2

# install python packages
ARG CPATH=/usr/local/cuda/targets/x86_64-linux/include
ARG CUDA_HOME=/usr/local/cuda
ARG LD_LIBRARY_PATH=/usr/local/cuda/lib64

COPY compile_pointnet_tfops.sh /env-setup/compile_pointnet_tfops.sh
# fix for build-time error: https://github.com/tensorflow/tensorflow/issues/10776#issuecomment-309128975
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
RUN cd /env-setup && \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs/:$LD_LIBRARY_PATH sh compile_pointnet_tfops.sh
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1

# for GUI
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt install -y \
    xserver-xorg-video-dummy \
    xfce4 desktop-base \
    x11vnc x11-apps net-tools
# disable screensaver
RUN apt autoremove -y xscreensaver
# optional: if you want a richer desktop experience
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    xfce4-terminal firefox
RUN echo 2 | update-alternatives --config x-terminal-emulator

RUN mkdir -p /opt/misc /opt/logs
COPY x-dummy.conf /opt/misc
COPY entrypoint.sh /opt/misc
RUN chmod +x /opt/misc/entrypoint.sh

ENV QT_X11_NO_MITSHM=1
ENV DISPLAY=:0

WORKDIR /code-dir
# ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "idsdf", "/bin/bash", "-c"]
ENTRYPOINT ["/opt/misc/entrypoint.sh"]
