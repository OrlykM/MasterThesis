FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

# Ubuntu 22.04 (not 20.04) matters: ORB_SLAM3's CMakeLists requires OpenCV > 4.4,
# and 20.04's apt libopencv-dev is only 4.2.0 -- 22.04's is 4.5.4.
RUN apt-get update && apt-get install -y \
    build-essential cmake git wget unzip pkg-config \
    libeigen3-dev libopencv-dev \
    libboost-dev libboost-serialization-dev \
    libgl1-mesa-dev libglew-dev \
    libpython3-dev python3-dev python3-numpy \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavutil-dev libavformat-dev libswscale-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Pangolin v0.6 -- known-compatible version with ORB_SLAM3's viewer code
RUN git clone --branch v0.6 --recursive https://github.com/stevenlovegrove/Pangolin.git /Pangolin && \
    cd /Pangolin && mkdir build && cd build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release && \
    make -j$(nproc) && make install && ldconfig

RUN git clone https://github.com/UZ-SLAMLab/ORB_SLAM3.git /ORB_SLAM3
WORKDIR /ORB_SLAM3
RUN chmod +x build.sh && ./build.sh

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]