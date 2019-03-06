# EDIT THE USER, WORKDIR, GROUPNAME names and IDs

FROM local/mm3-py3-jupyter:root

# remove inconvenient id
USER 0
RUN groupmod -g 200 dialout
RUN find / -gid 20 ! -type l -exec chgrp 200 {} \; 2>&1 > /dev/null | { grep -v 'No such file or directory' >&2; [ $? -eq 1 ];  }
RUN groupmod -g 20 staff && \
    find / -gid 50 ! -type l -exec chgrp 20 {} \; 2>&1 > /dev/null | { grep -v 'No such file or directory' >&2; [ $? -eq 1 ];  }

# install package for mac OSX socket sharing through x11
RUN apt-get install -y libglu1-mesa

# add new sudo user
ENV USERNAME jt
ENV GROUPNAME staff
RUN useradd -N -m $USERNAME --uid 504 --gid 20 && \
    echo "$USERNAME:$GROUPNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

USER jt
WORKDIR /home/jt
