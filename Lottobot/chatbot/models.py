from django.db import models
from django.conf import settings


class Chatbot(models.Model):
    author = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="chat_log")
    user_input = models.CharField(max_length=500)
    ai_response = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"User : {self.user_input} / AI : {self.ai_response}"




