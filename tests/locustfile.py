from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    host = "http://127.0.0.1:8080"
    wait_time = between(5, 15)
    
    @task
    def index(self):
        self.client.post(
            "/gradio_api/call/predict",
            {
                "data": [
                    "asdasd"
                ]
            }
        )
