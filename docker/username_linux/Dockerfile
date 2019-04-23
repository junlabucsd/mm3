# Edit the variables:
# _USERNAME_, _USERID_, _GROUPNAME_, and _GROUPID_ below

# For the Jupyter build, use local/mm3-py3-jupyter:root
FROM local/mm3-py3:root

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
