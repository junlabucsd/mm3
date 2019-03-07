# Edit the variables:
# _USERNAME_, _USERID_, _GROUPNAME_, and _GROUPID_ below

# For the Jupyter build, use local/mm3-py3-jupyter:root
FROM local/mm3-py3:root

# remove inconvenient id (Mac OSX)
USER 0
RUN groupmod -g 200 dialout # Change group 20 (dialout) to 200
RUN find / -gid 20 ! -type l -exec chgrp 200 {} \; 2>&1 > /dev/null | { grep -v 'No such file or directory' >&2; [ $? -eq 1 ];  }
RUN groupmod -g 20 staff && \
    find / -gid 50 ! -type l -exec chgrp 20 {} \; 2>&1 > /dev/null | { grep -v 'No such file or directory' >&2; [ $? -eq 1 ];  }

# Add new sudo user
# Edit these
ENV USERNAME _USERNAME_
ENV USERID _USERID_
ENV GROUPNAME _GROUPNAME_
ENV GROUPID _GROUPID_

# Don't edit this, it uses the variables above
RUN useradd -N -m $USERNAME --uid $USERID --gid $GROUPID && \
    echo "$USERNAME:$GROUPNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Edit these
USER _USERNAME_
WORKDIR /home/_USERNAME_
