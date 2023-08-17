# Define custom function directory
ARG FUNCTION_DIR="/function"

FROM public.ecr.aws/docker/library/python:buster as build-image

# Include global arg in this stage of the build
ARG FUNCTION_DIR
ENV USE_NNPACK=0
ENV USE_CUDA=0

# Install aws-lambda-cpp build dependencies
RUN apt-get update && \
  apt-get install -y \
  ffmpeg \
  libxext6 \
  libsm6

# Copy function code
RUN mkdir -p ${FUNCTION_DIR}
COPY . ${FUNCTION_DIR}
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install the function's dependencies
RUN pip install \
  --target ${FUNCTION_DIR} \
  -r requirements.txt


FROM public.ecr.aws/docker/library/python:buster

# Include global arg in this stage of the build
ARG FUNCTION_DIR
# Set working directory to function root directory
WORKDIR ${FUNCTION_DIR}

# Copy in the built dependencies
COPY --from=build-image ${FUNCTION_DIR} ${FUNCTION_DIR}

ENTRYPOINT [ "/usr/local/bin/python", "-m", "awslambdaric" ]
CMD [ "lambda_function.handler" ]