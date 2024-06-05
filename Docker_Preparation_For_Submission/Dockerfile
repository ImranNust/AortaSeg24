FROM --platform=linux/amd64 pytorch/pytorch


ENV PYTHONUNBUFFERED 1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Add the directory containing the scripts to PATH
ENV PATH="/home/user/.local/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"


COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user resources /opt/app/resources


RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
