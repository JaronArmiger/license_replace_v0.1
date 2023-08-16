FROM public.ecr.aws/lambda/python:3.9
RUN yum clean all
RUN yum update -y
RUN yum -y install epel-release
RUN yum install -y ffmpeg libsm6 libxext6
COPY requirements.txt ${LAMBDA_TASK_ROOT}
COPY . ${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt

CMD ["lambda_function.handler"]