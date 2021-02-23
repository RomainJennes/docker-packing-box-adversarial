# +--------------------------------------------------------------------------------------------------------------------+
# |                                         CREATE THE BOX BASED ON UBUNTU                                             |
# +--------------------------------------------------------------------------------------------------------------------+
FROM ubuntu:20.04 AS base
MAINTAINER Alexandre DHondt <alexandre.dhondt@gmail.com>
LABEL version="1.0.1"
LABEL source="https://github.com/dhondta/packing-box"
ENV DEBIAN_FRONTEND noninteractive
ENV TERM xterm-256color
# apply upgrade
RUN (apt -qq update \
 && apt -qq -y upgrade \
 && apt -qq -y autoremove \
 && apt -qq autoclean) 2>&1 > /dev/null \
 || echo -e "\033[1;31m SYSTEM UPGRADE FAILED \033[0m"
# install common dependencies, libraries and tools
RUN (apt -qq -y install apt-transport-https apt-utils \
 && apt -qq -y install bash-completion build-essential cmake software-properties-common \
 && apt -qq -y install libavcodec-dev libavformat-dev libavresample-dev libavutil-dev libbsd-dev libboost-regex-dev \
                   libboost-program-options-dev libboost-system-dev libboost-filesystem-dev libc6-dev-i386 \
                   libcairo2-dev libdbus-1-dev libegl1-mesa-dev libelf-dev libffi-dev libfontconfig1-dev \
                   libfreetype6-dev libfuse-dev libgif-dev libgirepository1.0-dev libgl1-mesa-dev libglib2.0-dev \
                   libglu1-mesa-dev libjpeg-dev libpulse-dev libssl-dev libsvm-java libtiff5-dev libudev-dev \
                   libxcursor-dev libxkbfile-dev libxml2-dev libxrandr-dev  \
 && apt -qq -y install colordiff colortail dosbox git golang less ltrace strace sudo tmate tmux vim xterm \
 && apt -qq -y install iproute2 nodejs npm python3-setuptools python3-pip swig weka x11-apps xvfb yarnpkg zstd \
 && apt -qq -y install curl unrar unzip wget) 2>&1 > /dev/null \
 || echo -e "\033[1;31m DEPENDENCIES INSTALL FAILED \033[0m"
# install wine (for running Windows software on Linux)
RUN (dpkg --add-architecture i386 \
 && wget -nc https://dl.winehq.org/wine-builds/winehq.key \
 && apt-key add winehq.key \
 && add-apt-repository 'deb https://dl.winehq.org/wine-builds/ubuntu/ focal main' \
 && apt -qq update \
 && apt -qq -y install --install-recommends winehq-stable wine32 \
 && wineboot) 2>&1 > /dev/null \
 || echo -e "\033[1;31m WINE INSTALL FAILED \033[0m"
# install mono (for running .NET apps on Linux)
RUN (apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF \
 && apt-add-repository 'deb https://download.mono-project.com/repo/ubuntu stable-focal main' \
 && apt -qq update \
 && apt -qq -y install mono-complete) 2>&1 > /dev/null \
 || echo -e "\033[1;31m MONO INSTALL FAILED \033[0m"
# install darling (for running MacOS software on Linux)
#RUN (apt -qq -y install cmake clang bison flex pkg-config linux-headers-generic gcc-multilib \
# && cd /tmp/ && git clone --recursive https://github.com/darlinghq/darling.git && cd darling \
# && mkdir build && cd build && cmake .. && make && make install \
# && make lkm && make lkm_install) 2>&1 > /dev/null \
# || echo -e "\033[1;31m DARLING INSTALL FAILED \033[0m"
# install/update Python packages
RUN (pip3 install pandas sklearn tinyscript weka \
 && pip3 freeze - local | grep -v "^\-e" | cut -d = -f 1 | xargs -n1 pip3 install -qU) 2>&1 > /dev/null \
 || echo -e "\033[1;31m PIP PACKAGES UPDATE FAILED \033[0m"
# +--------------------------------------------------------------------------------------------------------------------+
# |                     CUSTOMIZE THE BOX (refine the terminal, add folders to PATH and some aliases)                  |
# +--------------------------------------------------------------------------------------------------------------------+
FROM base AS customized
# copy customized files
ADD files/term/bash_aliases /root/.bash_aliases
ADD files/term/bash_colors /root/.bash_colors
ADD files/term/bash_gitprompt /root/.bash_gitprompt
ADD files/term/bashrc /root/.bashrc
ADD files/term/profile /root/.profile
ADD files/term/viminfo /root/.viminfo
# set the base files and folders for further setup
ADD *.yml /opt/
RUN mkdir -p /mnt/share /tmp/detectors /tmp/packers /tmp/unpackers /opt/detectors /opt/packers /opt/tools /opt/unpackers
# +--------------------------------------------------------------------------------------------------------------------+
# |                           ADD UTILITIES (that are not packers, unpackers or home-made tools)                       |
# +--------------------------------------------------------------------------------------------------------------------+
FROM customized AS utils
# copy pre-built utils
ADD files/utils/* /usr/bin/
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                    ADD TOOLS                                                       |
# +--------------------------------------------------------------------------------------------------------------------+
FROM utils AS tools
ADD files/tools/* /opt/tools/
ADD files/lib /tmp/lib
RUN pip3 install /tmp/lib/ 2>&1 > /dev/null \
 && mv /opt/tools/help /opt/tools/?
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                  ADD DETECTORS                                                     |
# +--------------------------------------------------------------------------------------------------------------------+
FROM tools AS detectors
ADD files/detectors/* /tmp/
RUN /opt/tools/packing-box setup detector
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                  ADD UNPACKERS                                                     |
# +--------------------------------------------------------------------------------------------------------------------+
FROM detectors AS unpackers
#ADD files/unpackers/* /tmp/
#RUN /opt/tools/packing-box setup unpacker
# +--------------------------------------------------------------------------------------------------------------------+
# |                                                   ADD PACKERS                                                      |
# +--------------------------------------------------------------------------------------------------------------------+
FROM unpackers AS packers
ADD files/packers/* /tmp/
RUN /opt/tools/packing-box setup packer
# ----------------------------------------------------------------------------------------------------------------------
ENTRYPOINT /opt/tools/startup
WORKDIR /mnt/share
