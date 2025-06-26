# Use the official TensorFlow image with the appropriate version
FROM tensorflow/tensorflow:2.16.1

# Set environment variables to prevent warnings from Streamlit
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=UTF-8

# Se t the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache for dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app directory into the container
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "hawaii_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
