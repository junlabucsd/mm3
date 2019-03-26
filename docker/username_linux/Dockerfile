# Edit the variables:
# _USERNAME_, _USERID_, _GROUPNAME_, and _GROUPID_ below

# For the Jupyter build, use local/mm3-py3-jupyter:root
FROM local/mm3-py3:root

# Add new sudo user
# Edit these
ENV USERNAME wanglab
ENV USERID 1000
ENV GROUPNAME wanglab
ENV GROUPID 1000

# Don't edit this, it uses the variables above
RUN useradd -N -m $USERNAME --uid $USERID --gid $GROUPID && \
    echo "$USERNAME:$GROUPNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME

# Edit these
USER wanglab
WORKDIR /home/wanglab/sandbox
