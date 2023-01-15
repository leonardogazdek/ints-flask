FROM python:3.7-slim

WORKDIR /app
COPY requirements.txt .
#nevjerojatno da tensorflow/tensorflow image
#jos uvijek ne podrzava arm64 sluzbeno
#tako da tf moramo instalirat preko pipa
#ako mislimo deployat na arm serveru
RUN pip install -r requirements.txt

COPY api.py .

CMD [ "python", "api.py" ]
